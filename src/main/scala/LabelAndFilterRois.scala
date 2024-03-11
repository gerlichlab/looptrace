package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import upickle.default.*
import cats.*
import cats.data.*
import cats.syntax.all.*
import mouse.boolean.*

import scopt.OParser
import com.github.tototoshi.csv.*

import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.*

/**
 * Measure data across all timepoints in the regions identified during spot detection.
 * 
 * Filter out spots that are too close together (e.g., because disambiguation 
 * of spots from indiviudal FISH probes in each region would be impossible in a multiplexed 
 * experiment).
 * 
 * If doing proximity-based filtration of spots, the value for the `"semantic"` key in the config 
 * file in the `"regionGrouping"` section should be either `"permissive"` or `"prohibitive"`. 
 * To treat all regional spots as a single group and prohibit proximal spots, the `"regionGrouping"` 
 * section may be omitted. To do 'no' proximity-based filtration, the minimum spot distance \
 * could be set to 0. 
 * 
 * For prohibitions, the groupings are interpreted by the program to mean that spots from the 
 * grouped regional barcodes may 'not' violate the proximity threshold; spots from other pairs 
 * of regional barcodes 'are' permitted to violate that threshold. 
 * For permissions, spots from pairs of regional barcodes in the same group 'may' violate 
 * the threshold while all others may 'not'.
 * 
 * For more, see related discussions on Github:
 * [[https://github.com/gerlichlab/looptrace/issues/71 Original, prohibitive semantics]]
 * [[https://github.com/gerlichlab/looptrace/issues/198 Newer, permissive semantics]]
 * [[https://github.com/gerlichlab/looptrace/issues/215 How to declare imaging rounds]]
 * [[https://github.com/gerlichlab/looptrace/pull/267 First config redesign]]
 * 
 * @author Vince Reuter
 */
object LabelAndFilterRois:
    val ProgramName = "LabelAndFilterRois"

    /**
      * The command-line configuration/interface definition
      *
      * @param spotsFile Path to the regional spots file in which to label records as too proximal or not
      * @param driftFile Path to the file with drift correction information for all ROIs, across all timepoints
      * @param probeGroups Specification of which regional barcode probes 
      * @param minSpotSeparation Number of units of separation required for points to not be considered proximal
      * @param buildDistanceThreshold How to use the minSpotSeparation value
      * @param unfilteredOutputFile Path to which to write unfiltered (but proximity-labeled) output
      * @param filteredOutputFile Path to which to write proximity-filtered (unlabeled) output
      * @param extantOutputHandler How to handle if an output target already exists
      */
    final case class CliConfig(
        configuration: ImagingRoundsConfiguration = null, // unconditionally required
        spotsFile: os.Path = null, // unconditionally required
        driftFile: os.Path = null, // unconditionally required
        spotSeparationThresholdValue: NonnegativeReal = NonnegativeReal(0), // unconditionally required
        buildDistanceThreshold: NonnegativeReal => DistanceThreshold = null, // unconditionally required
        unfilteredOutputFile: UnfilteredOutputFile = null, // unconditionally required
        filteredOutputFile: FilteredOutputFile = null, // unconditionally required
        extantOutputHandler: ExtantOutputHandler = null, // unconditionally required
        )

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import ScoptCliReaders.given
        import parserBuilder.*

        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, VersionName), 
            opt[ImagingRoundsConfiguration]("configuration")
                .required()
                .action((progConf, cliConf) => cliConf.copy(configuration = progConf))
                .text("Path to file specifying the imaging rounds configuration"),
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
            case Some(opts) => workflow(
                spotsFile = opts.spotsFile, 
                driftFile = opts.driftFile, 
                proximityFilterStrategy = opts.configuration.proximityFilterStrategy, 
                unfilteredOutputFile = opts.unfilteredOutputFile,
                filteredOutputFile = opts.filteredOutputFile, 
                extantOutputHandler = opts.extantOutputHandler,
                )
        }
    }

    def workflow(
        spotsFile: os.Path, 
        driftFile: os.Path, 
        proximityFilterStrategy: ImagingRoundsConfiguration.ProximityFilterStrategy,
        unfilteredOutputFile: UnfilteredOutputFile, 
        filteredOutputFile: FilteredOutputFile, 
        extantOutputHandler: ExtantOutputHandler
        ): Unit = {
        require(
            unfilteredOutputFile.toIO.getPath =!= filteredOutputFile.toIO.getPath, 
            s"Unfiltered and filtered outputs match: ${(unfilteredOutputFile, filteredOutputFile)}"
        )
        
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
                        s"${errors.length} error(s) converting spot file (${spotsFile}) rows to ROIs! First one: $h"
                        )
                }
            )
        }
        
        /* Then, parse the drift correction records from the corresponding file. */
        val driftByPosTimePair = {
            val drifts = withCsvData(driftFile){
                (driftRows: Iterable[CsvRow]) => Alternative[List].separate(driftRows.toList.map(rowToDriftRecord)) match {
                    case (Nil, drifts) => drifts
                    case (errors@(h :: _), _) => throw new Exception(
                        s"${errors.length} error(s) converting drift file (${driftFile}) rows to records! First one: $h"
                        )
                }
            }.asInstanceOf[List[DriftRecord]]

            val (recordNumbersByKey, keyed) = 
                NonnegativeInt.indexed(drifts)
                    .foldLeft(Map.empty[DriftKey, NonEmptySet[LineNumber]] -> Map.empty[DriftKey, DriftRecord]){ 
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
                    case ((PositionName(p), Timepoint(t)), lineNums) => (p, t) -> lineNums.toList.sorted
                }.sortBy(_._1)
                throw new Exception(s"${simpleReps.length} repeated (pos, time) pairs: ${simpleReps}")
            }
            keyed
        }

        /* For each ROI (by line number), look up its (too-proximal, according to distance threshold) neighbors. */
        // TODO: allow empty grouping here. https://github.com/gerlichlab/looptrace/issues/147
        val lookupNeighbors: LineNumber => Option[NonEmptySet[LineNumber]] = proximityFilterStrategy match {
            case ImagingRoundsConfiguration.UniversalProximityPermission => Function.const(None)
            case strategy: ImagingRoundsConfiguration.NontrivialProximityFilter =>
                val shiftedRoisNumbered = rowRoiPairs.map{ case ((_, oldRoi), idx) => 
                    val posTimePair = oldRoi.position -> oldRoi.time
                    val drift = driftByPosTimePair.getOrElse(posTimePair, throw new DriftRecordNotFoundError(posTimePair))
                    val newRoi = applyDrift(oldRoi, drift)
                    newRoi -> idx
                }
                buildNeighboringRoisFinder(shiftedRoisNumbered, strategy) match {
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

    final case class DriftRecordNotFoundError(key: DriftKey) extends NoSuchElementException(s"key not found: ($key)")

    /** Add the given total drift (coarse + fine) to the given ROI, updating its centroid and its bounding box accordingly. */
    private def applyDrift(roi: Roi, drift: DriftRecord): Roi = {
        require(
            roi.position === drift.position && roi.time === drift.time, 
            s"ROI and drift don't match on (FOV, time): (${roi.position -> roi.time} and (${drift.position -> drift.time})"
            )
        roi.copy(centroid = Movement.addDrift(drift.total)(roi.centroid), boundingBox = Movement.addDrift(drift.total)(roi.boundingBox))
    }

    /**
      * Create a mapping from item to other items within given distance threshold of that item.
      * 
      * An item is not considered its own neigbor, and this is omitting from the result any item with no proximal neighbors.
      *
      * @tparam B 
      * @tparam A The type items of interest, for which to compute sets of proximal neighbors
      * @tparam K The type of value on which to group the items
      * @param kvPairs The pairs of keying value and actual item
      * @param usePair Whether to consider a given pair of items for proximity, otherwise skip
      * @param getPoint How to get a point in 3D space from each arbitrary item
      * @param minDist The threshold distance by which to define two points as proximal
      * @param simplify How to get from a more complex value `B` to a simpler `A`
      * @return A mapping from item to set of proximal (closer than given distance threshold) neighbors, omitting each item with no neighbors
      */
    private[looptrace] def buildNeighborsLookupKeyed[B, A : Order, K : Order](
        kvPairs: Iterable[(K, B)], usePair: (B, B) => Boolean, getPoint: A => Point3D, minDist: DistanceThreshold, simplify: B => A
        ): Map[A, NonEmptySet[A]] = 
        import ProximityComparable.proximal
        given proxComp: ProximityComparable[A] = DistanceThreshold.defineProximityPointwise(minDist)(getPoint)
        // In the reduction, we don't care if on collision the Semigroup instance for Map will combine values or will overwrite them, 
        // since the partition on keys here guarantees no collisions between resulting submaps that are being combined.
        kvPairs.groupBy(_._1)
            .values // Keys are only used to create the partitions within which to compute distances, so discard after grouping.
            .map(_.map(_._2))
            .map{ part => 
                // Find all pairs of values more within the threshold of proximity w.r.t. one another.
                val closePairs: List[(A, A)] = part.toList.combinations(2).flatMap{
                    case b1 :: b2 :: Nil => usePair(b1, b2)
                        .option(simplify(b1) -> simplify(b2))
                        .flatMap{ (a1, a2) => (a1 `proximal` a2).option(a1 -> a2) }
                    case xs => throw new Exception(s"${xs.length} (not 2) element(s) when taking pairwise combinations!")
                }.toList
                // BIDIRECTIONALLY add the pair to the mapping, as we need the redundancy in order to remove BOTH spots.
                // https://github.com/gerlichlab/looptrace/issues/148
                closePairs.foldLeft(Map.empty[A, NonEmptySet[A]]){ case (acc, (a1, a2)) => 
                    val v1 = acc.get(a1).fold(NonEmptySet.one[A])(_.add)(a2)
                    val v2 = acc.get(a2).fold(NonEmptySet.one[A])(_.add)(a1)
                    acc ++ Map(a1 -> v1, a2 -> v2)
                }
            } // Compute the neighbors mapping within each part.
            .toSeq.combineAll // Combine all the partitioned neighbors mappings.

    /**
      * Create a mapping from record number (ROI) to set of record numbers encoding neighboring ROIs.
      * 
      * A ROI is not considered its own neigbor, and this is omitting from the result any item with no proximal neighbors.
      * 
      * @param rois Pair of {@code Roi} and its record number (0-based) from a CSV file
      * @param proxFilterStrategy Strategy with which to consider points as proximal ("neighbors") or not
      * @return A mapping from item to set of proximal (closer than given distance threshold) neighbors, omitting each item with no neighbors; 
      *         otherwise, a {@code Left}-wrapped error message about what went wrong
      * @see 
      */
    private[looptrace] def buildNeighboringRoisFinder(rois: List[RoiLinenumPair], proxFilterStrategy: ImagingRoundsConfiguration.NontrivialProximityFilter): Either[String, Map[LineNumber, NonEmptySet[LineNumber]]] = {
        given orderForRoiLinenumPair: Order[RoiLinenumPair] = Order.by(_._2)
        type GroupId = NonnegativeInt
        type IndexedRoi = (RegionalBarcodeSpotRoi, NonnegativeInt)
        val minSep = PiecewiseDistance.ConjunctiveThreshold(proxFilterStrategy.minSpotSeparation.asNonnegative)
        val getPoint = (_: RoiLinenumPair)._1.centroid
        val groupedRoiLinenumPairs: Either[String, Map[RoiLinenumPair, NonEmptySet[RoiLinenumPair]]] = proxFilterStrategy match {
            case _: ImagingRoundsConfiguration.UniversalProximityProhibition => 
                buildNeighborsLookupKeyed(rois.map{ case pair@(r, _) => r.position -> pair }, (_, _) => true, getPoint, minSep, identity).asRight
            case strat: (ImagingRoundsConfiguration.SelectiveProximityProhibition | ImagingRoundsConfiguration.SelectiveProximityPermission) => 
                val groupIds = NonnegativeInt.indexed(strat.grouping.toList).flatMap((g, i) => g.toList.map(_ -> i)).toMap
                Alternative[List].separate(rois.map{ case pair@(roi, _) => groupIds.get(roi.time).toRight(pair).map(_ -> pair)}) match {
                    case (Nil, withGroupsAssigned) => 
                        val considerRoiPair = strat match {
                            case _: ImagingRoundsConfiguration.SelectiveProximityProhibition => { (a: (GroupId, IndexedRoi), b: (GroupId, IndexedRoi)) => a._1 === b._1 }
                            case _: ImagingRoundsConfiguration.SelectiveProximityPermission => { (a: (GroupId, IndexedRoi), b: (GroupId, IndexedRoi)) => a._1 =!= b._1 }
                        }
                        buildNeighborsLookupKeyed(withGroupsAssigned.map{ case (gi, pair@(roi, _)) => roi.position -> (gi -> pair) }, considerRoiPair, getPoint, minSep, _._2).asRight
                    case (groupless, _) => 
                        val times = groupless.map(_._1.time).toSet
                        val timesText = times.toList.map(_.get).sorted.mkString(", ")
                        s"${groupless.length} ROIs without timepoint declared in grouping. ${times.size} undeclared timepoints: $timesText".asLeft
                }
        }
        groupedRoiLinenumPairs.map(_.map{ case ((_, idx), indexedNeighbors) => idx -> indexedNeighbors.map(_._2) })
    }

    /** Parse a {@code ROI} value from an in-memory representation of a single line from a CSV file. */
    def rowToRoi(row: CsvRow): ErrMsgsOr[Roi] = {
        val indexNel = safeGetFromRow("", safeParseInt >>> RoiIndex.fromInt)(row)
        val posNel = safeGetFromRow("position", PositionName.apply(_).asRight)(row)
        val regionNel = safeGetFromRow("frame", safeParseInt >>> RegionId.fromInt)(row)
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
        (indexNel, posNel, regionNel, channelNel, centroidNel, bboxNel)
            .mapN(RegionalBarcodeSpotRoi.apply)
            .toEither
    }
    
    /** 
     * Try to arse a single line of a CSV file to a representation of a drift correction record.
     * 
     * @param row The simply-parsed (all {@code String}) representation of a line from a CSV
     * @return Either a {@code Left}-wrapped nonempty collection of error messages, or a {@code Right}-wrapped record
     */
    private def rowToDriftRecord(row: CsvRow): ErrMsgsOr[DriftRecord] = {
        val posNel = safeGetFromRow("position", PositionName.apply(_).asRight)(row)
        val timeNel = safeGetFromRow("frame", safeParseInt >>> Timepoint.fromInt)(row)
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

    /* Type aliases */
    type DriftKey = (PositionName, Timepoint)
    type LineNumber = NonnegativeInt
    type PosInt = PositiveInt
    type Roi = RegionalBarcodeSpotRoi
    type RoiLinenumPair = (Roi, NonnegativeInt)
    
    /* Distinguish, at the type level, the semantic meaning of each output target. */
    opaque type FilteredOutputFile <: os.Path = os.Path
    object FilteredOutputFile:
        def fromPath(p: os.Path): FilteredOutputFile = p : FilteredOutputFile
    opaque type UnfilteredOutputFile <: os.Path = os.Path
    object UnfilteredOutputFile:
        def fromPath(p: os.Path): UnfilteredOutputFile = p : UnfilteredOutputFile

    extension [A](rw: ReadWriter[List[A]])
        def toNel(context: String): ReadWriter[NonEmptyList[A]] = 
            rw.bimap(_.toList, _.toNel.getOrElse{ throw new Exception(s"$context: No elements to read as nonempty list!") })

    /** Push a value through to the right side of an {@code Either}, pairing with the wrapped value */
    extension [A, L, R](f: A => Either[L, R])
        def throughRight: A => Either[L, (A, R)] = a => f(a).map(a -> _)

end LabelAndFilterRois
