package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ Failure, Success, Try }
import upickle.default.*
import cats.{ Alternative, Eq, Functor, Monoid, Order }
import cats.data.{ NonEmptyList as NEL, NonEmptyMap, NonEmptySet, ValidatedNel }
import cats.instances.tuple.*
import cats.syntax.all.*
import mouse.boolean.*

import scopt.{ OParser }
import com.github.tototoshi.csv.*

import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.{ fromJsonThruInt, safeExtract, readJsonFile }
import at.ac.oeaw.imba.gerlich.looptrace.space.{
    Coordinate, 
    DistanceThreshold, 
    EuclideanDistance, 
    PiecewiseDistance, 
    Point3D, ProximityComparable, 
    XCoordinate, 
    YCoordinate, 
    ZCoordinate
}
import at.ac.oeaw.imba.gerlich.looptrace.syntax.*

/**
 * Measure data across all timepoints in the regions identified during spot detection.
 * 
 * Optionally, also filter out spots that are too close together (e.g., because disambiguation 
 * of spots from indiviudal FISH probes in each region would be impossible in a multiplexed 
 * experiment).
 */
object LabelAndFilterRois:
    val ProgramName = "LabelAndFilterRois"

    /**
      * The command-line configuration/interface definition
      *
      * @param spotsFile Path to the (enriched, filtered) traces file to label and filter
      * @param driftFile Path to the file with drift correction information for all ROIs, across all timepoints
      * @param probeGroups Specification of which regional barcode probes 
      * @param minSpotSeparation Number of units of separation required for points to not be considered proximal
      * @param buildDistanceThreshold How to use the minSpotSeparation value
      * @param unfilteredOutputFile Path to which to write unfiltered (but proximity-labeled) output
      * @param filteredOutputFile Path to which to write proximity-filtered (unlabeled) output
      * @param extantOutputHandler How to handle if an output target already exists
      */
    final case class CliConfig(
        spotsFile: os.Path = null, // unconditionally required
        driftFile: os.Path = null, // unconditionally required
        probeGroups: List[ProbeGroup] = null, // unconditionally required
        spotSeparationThresholdValue: NonnegativeReal = NonnegativeReal(0), // unconditionally required
        buildDistanceThreshold: NonnegativeReal => DistanceThreshold = null, // unconditionally required
        unfilteredOutputFile: UnfilteredOutputFile = null, // unconditionally required
        filteredOutputFile: FilteredOutputFile = null, // unconditionally required
        extantOutputHandler: ExtantOutputHandler = null, // unconditionally required
        )

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import ScoptCliReaders.given
        import ThresholdSemantic.given
        import parserBuilder.*

        /** Parse a direct spec of an empty list, content of JSON file path, or CLI-arg-like mapping spec of probe groups. */
        given readProbeGroups: scopt.Read[List[ProbeGroup]] = {
            def readAsEmpty[A]: String => Either[String, List[A]] = arg => arg match {
                case "NONE" => List.empty[A].asRight
                case _ => s"Cannot parse argument as specification of empty list: $arg".asLeft
            }
            val readAsFile = (s: String) => 
                Try{ readJsonFile[List[ProbeGroup]](summon[scopt.Read[os.Path]].reads(s)) }.toEither.leftMap(_.getMessage)
            val readAsMap = (s: String) => {
                Try{ summon[scopt.Read[Map[Int, Int]]].reads(s) }
                    .toEither
                    .leftMap(_.getMessage)
                    .flatMap(
                        _.toList.foldRight(List.empty[Int] -> Map.empty[Int, NEL[FrameIndex]]){
                            // finds negative values and groups the probes (key) by group (value), inverting the mapping and collecting
                            case ((probe, groupIndex), (neg, acc)) => FrameIndex.fromInt(probe).fold(
                                _ => (probe :: neg, acc),
                                frame => (neg, acc + (groupIndex -> acc.get(groupIndex).fold(NEL.one(frame))(frame :: _)))
                                )
                        } match {
                            case (Nil, rawGroups) => 
                                val (histogram, groups) = rawGroups.values.foldRight(Map.empty[NonnegativeInt, Int] -> List.empty[ProbeGroup]){
                                    case (g, (hist, gs)) => {
                                        val subHist = g.map(_.get).groupBy(identity).view.mapValues(_.size).toMap
                                        (hist |+| subHist, ProbeGroup(g.toNes) :: gs)
                                    }
                                }
                                histogram.filter(_._2 > 1).toList.toNel.toLeft(groups).leftMap{ reps => s"Repeated frames in grouping: $reps" }
                            case (negatives, _) => s"Negative frame(s) in grouping: $negatives".asLeft
                        }
                    )
            }
            scopt.Read.reads{ arg => 
                // Preserve the first successful result and return it, otherwise accumulate the error messages.
                // So start with an empty list as a Left, then add error messages while staying in Left, or preserving the value in Right.
                // TODO: try to replace this with a defined structure, something like Validated + Alternative in Haskell.
                List(readAsMap, readAsFile, readAsEmpty).foldRight(Either.left[List[String], List[ProbeGroup]](List.empty[String])){ 
                    case (safeRead, maybeResult) => maybeResult.leftFlatMap(errors => safeRead(arg).leftMap(_ :: errors))
                }.fold(msgs => throw new IllegalArgumentException(s"Cannot parse arg ($arg) as probe grouping. Errors: $msgs"), identity)
            }
        }

        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, VersionName), 
            opt[os.Path]("spotsFile")
                .required()
                .action((f, c) => c.copy(spotsFile = f))
                .validate(f => os.isFile(f).either(f"Alleged spots file isn't a file: $f", ()))
                .text("Path to regional spots file"),
            opt[os.Path]("driftFile")
                .required()
                .action((f, c) => c.copy(driftFile = f))
                .validate(f => os.isFile(f).either(f"Alleged drift file isn't a file: $f", ()))
                .text("Path to drift correction file"),
            opt[List[ProbeGroup]]("probeGroups")
                .required()
                .action((gs, c) => c.copy(probeGroups = gs))
                .text("Either mapping from probe ID to grouping ID, or path to JSON file (list of lists), or 'NONE' to indicate no grouping"),
            opt[NonnegativeReal]("spotSeparationThresholdValue")
                .required()
                .action((d, c) => c.copy(spotSeparationThresholdValue = d))
                .text("Min separation between centers of pair of spots, discard otherwise; contextualised by --spotSeparationThresholdType"),
            opt[NonnegativeReal => DistanceThreshold]("spotSeparationThresholdType")
                .required()
                .action((f, c) => c.copy(buildDistanceThreshold = f))
                .text("How to use the raw numeric value given for minimum spot separation"),
            opt[UnfilteredOutputFile]("unfilteredOutputFile")
                .required()
                .action((f, c) => c.copy(unfilteredOutputFile = f))
                .text("Path to file to which to write unfiltered output"),
            opt[FilteredOutputFile]("filteredOutputFile")
                .required()
                .action((f, c) => c.copy(filteredOutputFile = f))
                .text("Path to file to which to write filtered output"),
            opt[ExtantOutputHandler]("handleExtantOutput")
                .required()
                .action((h, c) => c.copy(extantOutputHandler = h))
                .text("How to handle writing output when target already exists"),
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => 
                val threshold = opts.buildDistanceThreshold(opts.spotSeparationThresholdValue)
                workflow(
                    spotsFile = opts.spotsFile, 
                    driftFile = opts.driftFile, 
                    probeGroups = opts.probeGroups, 
                    minSpotSeparation = threshold, 
                    unfilteredOutputFile = opts.unfilteredOutputFile,
                    filteredOutputFile = opts.filteredOutputFile, 
                    extantOutputHandler = opts.extantOutputHandler
                    )
        }
    }

    def workflow(
        spotsFile: os.Path, 
        driftFile: os.Path, 
        probeGroups: List[ProbeGroup],
        minSpotSeparation: DistanceThreshold, 
        unfilteredOutputFile: UnfilteredOutputFile, 
        filteredOutputFile: FilteredOutputFile, 
        extantOutputHandler: ExtantOutputHandler
        ): Unit = {
        
        /* Create unsafe CSV writer for each output type, failing fast if either output exists and overwrite is not active.
           In the process, bind each target output file to its corresponding named function. */
        val (writeUnfiltered, writeFiltered) = {
            val unfilteredNel = extantOutputHandler.getSimpleWriter(unfilteredOutputFile).toValidatedNel
            val filteredNel = extantOutputHandler.getSimpleWriter(filteredOutputFile).toValidatedNel
            val writeCsv = (sink: os.Source => Boolean) => writeAllCsvUnsafe(sink)(_: List[String], _: Iterable[CsvRow])
            (unfilteredNel, filteredNel).tupled.fold(
                es => throw new Exception(s"${es.size} existence error(s) with output: ${es.map(_.getMessage)}"), 
                _.mapBoth(writeCsv)
                )
        }

        /* Then, parse the ROI records from the (regional barcode) spots file. */
        val (roisHeader, rowRoiPairs): (List[String], List[((CsvRow, Roi), LineNumber)]) = {
            safeReadAllWithOrderedHeaders(spotsFile).fold(
                throw _, 
                (head, spotRows) => Alternative[List].separate(spotRows.map(rowToRoi.throughRight)) match {
                    case (Nil, rrPairs) => head -> NonnegativeInt.indexed(rrPairs)
                    case (errors@(h :: _), _) => throw new Exception(
                        s"${errors.length} errors converting spot file (${spotsFile}) rows to ROIs! First one: $h"
                        )
                }
            )
        }
        
        /* Then, parse the drift correction records from the corresponding file. */
        val drifts = withCsvData(driftFile){
            (driftRows: Iterable[CsvRow]) => Alternative[List].separate(driftRows.toList.map(rowToDriftRecord)) match {
                case (Nil, drifts) => drifts
                case (errors@(h :: _), _) => throw new Exception(
                    s"${errors.length} errors converting drift file (${driftFile}) rows to records! First one: $h"
                    )
            }
        }.asInstanceOf[List[DriftRecord]]
        val driftByPosTimePair = {
            type Key = (PositionName, FrameIndex)
            val (recordNumbersByKey, keyed) = 
                NonnegativeInt.indexed(drifts)
                    .foldLeft(Map.empty[Key, NonEmptySet[LineNumber]] -> Map.empty[Key, DriftRecord]){ 
                        case ((reps, acc), (drift, recnum)) =>  
                            val p = drift.position
                            val t = drift.time
                            val k = p -> t
                            reps.get(k) match {
                                case None => (reps + (k -> NonEmptySet.one(recnum)), acc + (k -> drift))
                                case Some(prevLineNums) => (reps + (k -> prevLineNums.add(recnum)), acc)
                            }
                    }
            val repeats = recordNumbersByKey.filter(_._2.size > 1)
            if (repeats.nonEmpty) { 
                val simpleReps = repeats.toList.map{ 
                    case ((PositionName(p), FrameIndex(t)), lineNums) => (p, t) -> lineNums.toList.sorted
                }.sortBy(_._1)
                throw new Exception(s"${simpleReps.length} repeated (pos, time) pairs: ${simpleReps}")
            }
            keyed
        }

        /* For each ROI (by line number), look up its (too-proximal, according to distance threshold) neighbors. */
        // TODO: allow empty grouping here. https://github.com/gerlichlab/looptrace/issues/147
        val lookupNeighbors: LineNumber => Option[NonEmptySet[LineNumber]] = {
            val shiftedRoisNumbered = rowRoiPairs.map{ case ((_, oldRoi), idx) => 
                val posTimePair = oldRoi.position -> oldRoi.time
                val newRoi = applyDrift(oldRoi, driftByPosTimePair(posTimePair))
                newRoi -> idx
            }
            buildNeighboringRoisFinder(shiftedRoisNumbered, minSpotSeparation)(probeGroups) match {
                case Left(errMsg) => throw new Exception(errMsg)
                case Right(neighborsByRecordNumber) => neighborsByRecordNumber.get
            }
        }

        val roiRecordsLabeled = rowRoiPairs.map{ 
            case ((row, _), linenum) => row -> lookupNeighbors(linenum).map(_.toNonEmptyList.sorted)
        }

        /* Write the unfiltered output and print out the header */
        val unfilteredHeader = {
            val neighborColumnName = "neighbors"
            val header = roisHeader :+ neighborColumnName
            val records = roiRecordsLabeled.map{ case (row, maybeNeighbors) => 
                row + (neighborColumnName -> maybeNeighbors.fold(List())(_.toList).mkString(MultiValueFieldInternalSeparator))
            }
            val wroteIt = writeUnfiltered(header, records)
            println(s"${if wroteIt then "Wrote" else "Did not write"} unfiltered output file: $filteredOutputFile")
            header
        }
        
        val wroteIt = writeFiltered(roisHeader, roiRecordsLabeled.filter(_._2.isEmpty).map(_._1))
        println(s"${if wroteIt then "Wrote" else "Did not write"} filtered output file: $filteredOutputFile")

    }

    /****************************************************************************************************************
     * Main types and business logic
     ****************************************************************************************************************/
    final case class DriftRecord(position: PositionName, time: Timepoint, coarse: CoarseDrift, fine: FineDrift):
        def total = 
            // For justification of additivity, see: https://github.com/gerlichlab/looptrace/issues/194
            TotalDrift(
                ZDir(coarse.z.get + fine.z.get), 
                YDir(coarse.y.get + fine.y.get), 
                XDir(coarse.x.get + fine.x.get)
                )

    /**
      * Designation of regional barcode frame/probe/timepoint indices which are prohibited from being in (configurably) close proximity.
      *
      * @param get The actual collection of indices
      */
    final case class ProbeGroup(get: NonEmptySet[FrameIndex])
    
    /** Helpers for working with frame/probe/timepoint index groupings */
    object ProbeGroup:
        given rwForProbeGroup: ReadWriter[ProbeGroup] = readwriter[ujson.Value].bimap(
            group => ujson.Arr(group.get.toList.map(name => ujson.Num(name.get))*), 
            json => json.arr
                .toList
                .toNel
                .toRight("Empty collection can't parse as probe group!")
                .flatMap(_.traverse(_.safeInt.flatMap(FrameIndex.fromInt)))
                .flatMap(safeNelToNes)
                .leftMap(repeats => s"Repeat values for probe group: $repeats")
                .fold(msg => throw new ujson.Value.InvalidData(json, msg), ProbeGroup.apply)
        )
    end ProbeGroup

    /** Add the given total drift (coarse + fine) to the given ROI, updating its centroid and its bounding box accordingly. */
    private def applyDrift(roi: Roi, drift: DriftRecord): Roi = {
        require(
            roi.position === drift.position && roi.time === drift.time, 
            s"ROI and drift don't match on (FOV, time): (${roi.position -> roi.time} and (${drift.position -> drift.time})"
            )
        roi.copy(centroid = Movement.addDrift(drift.total)(roi.centroid), boundingBox = Movement.addDrift(drift.total)(roi.boundingBox))
    }

    /**
      * Construct a mapping from single record to collection of proximal neighbors, for filtration
      *
      * @tparam A The type of key for an individual row/record
      * @param getPoint How to extract a 3D point from the key
      * @param things The collection in which to determine groups of pairwise proximity.
      * @param minDist The definition of proximity
      * @return A mapping from single item to nonempty set of all items neighbors; items with no proximal neigbors are absent
      */
    def buildNeighborsLookupFlat[A : Order, D](getPoint: A => Point3D)(things: Iterable[A], minDist: DistanceThreshold): Map[A, NonEmptySet[A]] = {
        import ProximityComparable.proximal
        given proxComp: ProximityComparable[A] = DistanceThreshold.defineProximityPointwise(minDist)(getPoint)
        // Find all pairs of values more within the threshold of proximity w.r.t. one another.
        val closePairs: List[(A, A)] = things.toList.combinations(2).flatMap{
            case a1 :: a2 :: Nil => (a1 `proximal` a2).option(a1 -> a2)
            case xs => throw new Exception(s"${xs.length} (not 2!) element(s) when taking pairwise combinations")
        }.toList
        // BIDIRECTIONALLY add the pair to the mapping, as we need the redundancy in order to remove BOTH spots.
        // https://github.com/gerlichlab/looptrace/issues/148
        closePairs.foldLeft(Map.empty[A, NonEmptySet[A]]){ case (acc, (a1, a2)) => 
            val v1 = acc.get(a1).fold(NonEmptySet.one[A])(_.add)(a2)
            val v2 = acc.get(a2).fold(NonEmptySet.one[A])(_.add)(a1)
            acc ++ Map(a1 -> v1, a2 -> v2)
        }
    }

    /**
      * Create a mapping from item to other items within given distance threshold of that item.
      * 
      * An item is not considered its own neigbor, and this is omitting from the result any item with no proximal neighbors.
      *
      * @tparam A The type items of interest, for which to compute sets of proximal neighbors
      * @tparam K The type of value on which to group the items
      * @param getPoint How to get a point in 3D space from each arbitrary item
      * @param kvPairs The pairs of keying value and actual item
      * @param minDist The threshold distance by which to define two points as proximal
      * @return A mapping from item to set of proximal (closer than given distance threshold) neighbors, omitting each item with no neighbors
      */
    def buildNeighborsLookupKeyed[A : Order, K : Order](
        getPoint: A => Point3D)(kvPairs: Iterable[(K, A)], minDist: DistanceThreshold
        ): Map[A, NonEmptySet[A]] = {
        // In the reduction, we don't care if on collision the Semigroup instance for Map will combine values or will overwrite them, 
        // since the partition on keys here guarantees no collisions between resulting submaps that are being combined.
        Monoid.combineAll(kvPairs.groupBy(_._1).values.map{ subKVs => buildNeighborsLookupFlat(getPoint)(subKVs.map(_._2), minDist) })
    }

    /**
      * Create a mapping from record number (ROI) to set of record numbers encoding neighboring ROIs.
      * 
      * A ROI is not considered its own neigbor, and this is omitting from the result any item with no proximal neighbors.
      * 
      * @param rois Pair of {@code Roi} and its record number (0-based) from a CSV file
      * @param minDist The threshold on distance, beneath which two points are to be considered proximal
      * @param grouping The grouping of regional barcode timepoint/frame/probe indices, optional
      * @return A mapping from item to set of proximal (closer than given distance threshold) neighbors, omitting each item with no neighbors; 
      *         otherwise, a {@code Left}-wrapped error message about what went wrong
      */
    def buildNeighboringRoisFinder(
        rois: List[RoiLinenumPair], minDist: DistanceThreshold)(grouping: List[ProbeGroup]
        ): Either[String, Map[LineNumber, NonEmptySet[LineNumber]]] = {
        given orderForRoiLinenumPair: Order[RoiLinenumPair] = Order.by(_._2)
        val getPoint = (_: RoiLinenumPair)._1.centroid
        val groupedRoiLinenumPairs: Either[String, Map[RoiLinenumPair, NonEmptySet[RoiLinenumPair]]] = 
            if grouping.isEmpty 
            then buildNeighborsLookupKeyed(getPoint)(rois.map{ case t@(r, _) => r.position -> t }, minDist).asRight
            else {
                val (groupIds, repeatedFrames) = NonnegativeInt.indexed(grouping)
                    .flatMap((g, i) => g.get.toList.map(_ -> i))
                    .foldLeft(Map.empty[FrameIndex, NonnegativeInt] -> Map.empty[FrameIndex, Int]){ 
                        case ((ids, repeats), (frame, gid)) =>
                            if ids `contains` frame
                            then (ids, repeats + (frame -> (repeats.getOrElse(frame, 0) + 1)))
                            else (ids + (frame -> gid), repeats)
                    }
                if (repeatedFrames.nonEmpty) // Probe groupings isn't a partition, because there's overlap between the declared equivalence classes.
                then s"${repeatedFrames.size} repeated frame(s): $repeatedFrames".asLeft
                else {
                    // TODO: check that the union of the (now checked as disjoint) equivalence classes 
                    //       implied by the probe groupings cover the set of regional barcodes frames.
                    val (groupless, keyedRois) = Alternative[List].separate(rois.map{ case pair@(roi, _) => 
                        groupIds.get(roi.time)
                            .toRight(pair)
                            .map(groupIndex => ((roi.position, groupIndex), pair))
                    })
                    groupless.isEmpty.either(
                        {
                            val times = groupless.map(_._1.time).toSet
                            s"${groupless.length} ROIs without value declared in grouping. ${times.size} undeclared timepoints: $times"
                        }, 
                        buildNeighborsLookupKeyed(getPoint)(keyedRois, minDist)
                        )
                }
            }
        groupedRoiLinenumPairs.map(_.map{ case ((_, idx), indexedNeighbors) => idx -> indexedNeighbors.map(_._2) })
    }

    /** Parse a {@code ROI} value from an in-memory representation of a single line from a CSV file. */
    def rowToRoi(row: CsvRow): ErrMsgsOr[Roi] = {
        val indexNel = safeGetFromRow("", safeParseInt >>> RoiIndex.fromInt)(row)
        val posNel = safeGetFromRow("position", PositionName.apply(_).asRight)(row)
        val timeNel = safeGetFromRow("frame", safeParseInt >>> FrameIndex.fromInt)(row)
        val channelNel = safeGetFromRow("ch", safeParseInt >>> Channel.fromInt)(row)
        val centroidNel = {
            val zNel = safeGetFromRow("zc", safeParseDouble >> ZCoordinate.apply)(row)
            val yNel = safeGetFromRow("yc", safeParseDouble >> YCoordinate.apply)(row)
            val xNel = safeGetFromRow("xc", safeParseDouble >> XCoordinate.apply)(row)
            (zNel, yNel, xNel).mapN((z, y, x) => Point3D(x, y, z))
        }
        val bboxNel = {
            val zMinNel = safeGetFromRow("z_min", safeParseDouble >> ZCoordinate.apply)(row)
            val zMaxNel = safeGetFromRow("z_max", safeParseDouble >> ZCoordinate.apply)(row)
            val yMinNel = safeGetFromRow("y_min", safeParseDouble >> YCoordinate.apply)(row)
            val yMaxNel = safeGetFromRow("y_max", safeParseDouble >> YCoordinate.apply)(row)
            val xMinNel = safeGetFromRow("x_min", safeParseDouble >> XCoordinate.apply)(row)
            val xMaxNel = safeGetFromRow("x_max", safeParseDouble >> XCoordinate.apply)(row)
            (zMinNel, zMaxNel, yMinNel, yMaxNel, xMinNel, xMaxNel).mapN(
                (zMin, zMax, yMin, yMax, xMin, xMax) => BoundingBox(
                    sideX = BoundingBox.Interval(xMin, xMax),
                    sideY = BoundingBox.Interval(yMin, yMax),
                    sideZ = BoundingBox.Interval(zMin, zMax)
                )
            )
        }
        (indexNel, posNel, timeNel, channelNel, centroidNel, bboxNel)
            .mapN(RegionalBarcodeSpotRoi.apply)
            .toEither
    }
    
    /** 
     * Try to arse a single line of a CSV file to a representation of a drift correction record.
     * 
     * @param row The simply-parsed (all {@code String}) representation of a line from a CSV
     * @return Either a {@code Left}-wrapped nonempty collection of error messages, or a {@code Right}-wrapped record
     */
    def rowToDriftRecord(row: CsvRow): ErrMsgsOr[DriftRecord] = {
        val posNel = safeGetFromRow("position", PositionName.apply(_).asRight)(row)
        val timeNel = safeGetFromRow("frame", safeParseInt >>> FrameIndex.fromInt)(row)
        val coarseDriftNel = {
            val zNel = safeGetFromRow("z_px_coarse", safeParseIntLike >> ZDir.apply)(row)
            val yNel = safeGetFromRow("y_px_coarse", safeParseIntLike >> YDir.apply)(row)
            val xNel = safeGetFromRow("x_px_coarse", safeParseIntLike >> XDir.apply)(row)
            (zNel, yNel, xNel).mapN(CoarseDrift.apply)
        }
        val fineDriftNel = {
            val zNel = safeGetFromRow("z_px_fine", safeParseDouble >> ZDir.apply)(row)
            val yNel = safeGetFromRow("x_px_fine", safeParseDouble >> YDir.apply)(row)
            val xNel = safeGetFromRow("y_px_fine", safeParseDouble >> XDir.apply)(row)
            (zNel, yNel, xNel).mapN(FineDrift.apply)
        }
        (posNel, timeNel, coarseDriftNel, fineDriftNel).mapN(DriftRecord.apply).toEither
    }

    /** Try to read an integer from a string, failing if it's non-numeric or if conversion to {@code Int} would be lossy. */
    def safeParseIntLike: String => Either[String, Int] = safeParseDouble >>> tryToInt

    /****************************************************************************************************************
     * Ancillary definitions
     ****************************************************************************************************************/

    // Delimiter between fields within a multi-valued field (i.e., multiple values in 1 CSV column)
    val MultiValueFieldInternalSeparator = "|"

    object Movement:
        def shiftBy(del: XDir[Double])(c: XCoordinate): XCoordinate = XCoordinate(c.get + del.get)
        def shiftBy(del: YDir[Double])(c: YCoordinate): YCoordinate = YCoordinate(c.get + del.get)
        def shiftBy(del: ZDir[Double])(c: ZCoordinate): ZCoordinate = ZCoordinate(c.get + del.get)
        def shiftBy(del: XDir[Double])(intv: BoundingBox.Interval[XCoordinate]): BoundingBox.Interval[XCoordinate] =
            BoundingBox.Interval(shiftBy(del)(intv.lo), shiftBy(del)(intv.hi))
        def shiftBy(del: YDir[Double])(intv: BoundingBox.Interval[YCoordinate]): BoundingBox.Interval[YCoordinate] =
            BoundingBox.Interval(shiftBy(del)(intv.lo), shiftBy(del)(intv.hi))
        def shiftBy(del: ZDir[Double])(intv: BoundingBox.Interval[ZCoordinate]): BoundingBox.Interval[ZCoordinate] =
            BoundingBox.Interval(shiftBy(del)(intv.lo), shiftBy(del)(intv.hi))
        def addDrift(drift: TotalDrift)(pt: Point3D): Point3D = 
            Point3D(shiftBy(drift.x)(pt.x), shiftBy(drift.y)(pt.y), shiftBy(drift.z)(pt.z))
        def addDrift(drift: TotalDrift)(box: BoundingBox): BoundingBox = 
            BoundingBox(shiftBy(drift.x)(box.sideX), shiftBy(drift.y)(box.sideY), shiftBy(drift.z)(box.sideZ))
    end Movement

    sealed trait Direction[A : Numeric] { def get: A }
    final case class XDir[A : Numeric](get: A) extends Direction[A]
    final case class YDir[A : Numeric](get: A) extends Direction[A]
    final case class ZDir[A : Numeric](get: A) extends Direction[A]

    sealed trait Drift[A : Numeric]:
        def z: ZDir[A]
        def y: YDir[A]
        def x: XDir[A]
    end Drift
    final case class CoarseDrift(z: ZDir[Int], y: YDir[Int], x: XDir[Int]) extends Drift[Int]
    final case class FineDrift(z: ZDir[Double], y: YDir[Double], x: XDir[Double]) extends Drift[Double]
    final case class TotalDrift(z: ZDir[Double], y: YDir[Double], x: XDir[Double]) extends Drift[Double]

    /** Helpers for working with spot separation semantic values */
    object ThresholdSemantic:
        /** How to parse a spot separation semantic from a command-line argument */
        given readForDistanceThresholdSemantic: scopt.Read[NonnegativeReal => DistanceThreshold] = scopt.Read.reads(
            (_: String) match {
                case ("EachAxisAND" | "PiecewiseAND") => PiecewiseDistance.ConjunctiveThreshold.apply
                case ("Euclidean" | "EUCLIDEAN" | "euclidean" | "eucl" | "EUCL" | "euc" | "EUC") => EuclideanDistance.Threshold.apply
                case s => throw new IllegalArgumentException(s"Cannot parse as threshold semantic: $s")
            }
        )
    end ThresholdSemantic
    
    /* Type aliases */
    type LineNumber = NonnegativeInt
    type PosInt = PositiveInt
    type Roi = RegionalBarcodeSpotRoi
    type RoiLinenumPair = (Roi, NonnegativeInt)
    type Timepoint = FrameIndex
    
    /* Distinguish, at the type level, the semantic meaning of each output target. */
    opaque type FilteredOutputFile <: os.Path = os.Path
    object FilteredOutputFile:
        def fromPath(p: os.Path): FilteredOutputFile = p : FilteredOutputFile
    opaque type UnfilteredOutputFile <: os.Path = os.Path
    object UnfilteredOutputFile:
        def fromPath(p: os.Path): UnfilteredOutputFile = p : UnfilteredOutputFile

    extension [A](rw: ReadWriter[List[A]])
        def toNel(context: String): ReadWriter[NEL[A]] = 
            rw.bimap(_.toList, _.toNel.getOrElse{ throw new Exception(s"$context: No elements to read as nonempty list!") })

    /** Push a value through to the right side of an {@code Either}, pairing with the wrapped value */
    extension [A, L, R](f: A => Either[L, R])
        def throughRight: A => Either[L, (A, R)] = a => f(a).map(a -> _)

end LabelAndFilterRois
