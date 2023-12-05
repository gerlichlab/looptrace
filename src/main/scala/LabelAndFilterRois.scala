package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ Failure, Success, Try }
import upickle.default.*
import cats.{ Alternative, Eq, Functor, Order }
import cats.data.{ NonEmptyList as NEL, NonEmptyMap, NonEmptySet, ValidatedNel }
import cats.data.Validated.Valid
import cats.syntax.all.*
import mouse.boolean.*

import scopt.{ OParser, Read }
import com.github.tototoshi.csv.*

import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.{ fromJsonThruInt, safeExtract, readJsonFile }
import at.ac.oeaw.imba.gerlich.looptrace.space.{ Coordinate, EuclideanDistance, Metric, Point3D, XCoordinate, YCoordinate, ZCoordinate }
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

    type LineNumber = NonnegativeInt
    type PosInt = PositiveInt
    type Timepoint = FrameIndex
    
    opaque type FilteredOutputFile <: os.Path = os.Path
    opaque type UnfilteredOutputFile <: os.Path = os.Path

    val MultiValueFieldInternalSeparator = "|"

    case class CliConfig(
        spotsFile: os.Path = null, // unconditionally required
        driftFile: os.Path = null, // unconditionally required
        probeGroups: List[ProbeGroup] = null, // unconditionally required
        minSpotSeparation: NonnegativeReal = NonnegativeReal(0), // unconditionally required
        unfilteredOutputFile: UnfilteredOutputFile = null, // unconditionally required
        filteredOutputFile: FilteredOutputFile = null, // unconditionally required
        extantOutputHandler: ExtantOutputHandler = null, // unconditionally required
        )

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import ScoptCliReaders.given
        import parserBuilder.*

        given readProbeGroups: Read[List[ProbeGroup]] = {
            def readAsEmpty[A]: String => Either[String, List[A]] = arg => arg match {
                case "NONE" => List.empty[A].asRight
                case _ => s"Cannot parse argument as specification of empty list: $arg".asLeft
            }
            val readAsFile = (s: String) => 
                Try{ readJsonFile[List[ProbeGroup]](summon[Read[os.Path]].reads(s)) }.toEither.leftMap(_.getMessage)
            val readAsMap = (s: String) => {
                Try{ summon[Read[Map[Int, Int]]].reads(s) }
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
            Read.reads{ s => 
                // Preserve the first successful result and return it, otherwise accumulate the error messages.
                // So start with an empty list as a Left, then add error messages while staying in Left, or preserving the value in Right.
                // TODO: try to replace this with a defined structure, something like Validated + Alternative in Haskell.
                List(readAsMap, readAsFile, readAsEmpty).foldRight(Either.left[List[String], List[ProbeGroup]](List.empty[String])){ 
                    case (safeRead, maybeResult) => maybeResult.leftFlatMap(errors => safeRead(s).leftMap(_ :: errors))
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
            opt[NonnegativeReal]("minSpotSeparation")
                .required()
                .action((px, c) => c.copy(minSpotSeparation = px))
                .text("Minimum number of pixels required between centroids of a pair of spots; discard otherwise"),
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
                probeGroups = opts.probeGroups, 
                minSpotSeparation = opts.minSpotSeparation, 
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
        minSpotSeparation: NonnegativeReal, 
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
            println(s"Reading ROIs file: $spotsFile")
            safeReadAllWithOrderedHeaders(spotsFile).fold(
                throw _, 
                (head, spotRows) => Alternative[List].separate(spotRows.map(rowToRoi.throughRight)) match {
                    case (Nil, rrPairs) => head -> NonnegativeInt.indexed(rrPairs)
                    case (errors@(h :: _), _) => throw new Exception(s"${errors.length} errors converting spot file (${spotsFile}) rows to ROIs! First one: $h")
                }
            )
        }
        
        /* Then, parse the drift correction records from the corresponding file. */
        println(s"Reading drift file: $driftFile")
        val drifts = withCsvData(driftFile){ (driftRows: Iterable[CsvRow]) => Alternative[List].separate(driftRows.toList.map(rowToDriftRecord)) match {
            case (Nil, drifts) => drifts
            case (errors@(h :: _), _) => throw new Exception(s"${errors.length} errors converting drift file (${driftFile}) rows to records! First one: $h")
        } }.asInstanceOf[List[DriftRecord]]
        println("Keying drifts")
        val driftByPosTimePair = {
            type Key = (String, FrameIndex)
            val (repeats, keyed) = NonnegativeInt.indexed(drifts).foldLeft(Map.empty[Key, NonEmptySet[LineNumber]] -> Map.empty[Key, DriftRecord]){ 
                case ((reps, acc), (drift, recnum)) =>  
                    val p = drift.position
                    val t = drift.time
                    val k = p -> t
                    if acc `contains` k 
                    then (reps + (k -> reps.get(k).fold(NonEmptySet.one[LineNumber])(_.add)(recnum)), acc)
                    else (reps, acc + (k -> drift))
            }
            if (repeats.nonEmpty) { throw new Exception(s"${repeats.size} repeated (pos, time) pairs: ${repeats}") }
            keyed
        }
        
        /* For each ROI (by line number), look up its (too-proximal, according to distance threshold) neighbors. */
        println("Building neighbors lookup")
        val lookupNeighbors: LineNumber => Option[NonEmptySet[LineNumber]] = (probeGroups.toNel match {
            case None => { (_: LineNumber) => None }.asRight
            case Some(groups) => 
                // Use the drift-corrected centroid of each ROI as the basis for evaluation of whether it's too proximal to other points.
                given met: Metric[(Roi, LineNumber), EuclideanDistance, EuclideanDistance.Threshold] = EuclideanDistance.getMetric{
                    case (roi, _) => 
                        val drift = driftByPosTimePair(roi.position -> roi.time)
                        shiftPoint(drift.coarse)(roi.centroid)
                }
                buildNeighboringRoisFinder(rowRoiPairs.map{ case ((_, roi), idx) => roi -> idx}, minSpotSeparation)(groups).map(_.get)
        }).fold(errMsg => throw new Exception(errMsg), identity)

        println("Pairing neighbors with ROIs")
        val roiRecordsLabeled = rowRoiPairs.map{ case ((row, _), linenum) => row -> lookupNeighbors(linenum).map(_.toNonEmptyList.sorted) }

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
        println(s"Unfiltered header: $unfilteredHeader")

        val wroteIt = writeFiltered(roisHeader, roiRecordsLabeled.filter(_._2.isEmpty).map(_._1))
        println(s"${if wroteIt then "Wrote" else "Did not write"} filtered output file: $filteredOutputFile")

        println("Done!")
    }

    /****************************************************************************************************************
     * Main types and business logic
     ****************************************************************************************************************/
    final case class DriftRecord(position: String, time: Timepoint, coarse: CoarseDrift, fine: FineDrift)

    final case class ProbeGroup(get: NonEmptySet[FrameIndex])
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

    final case class Roi(index: RoiIndex, position: String, time: FrameIndex, channel: Channel, centroid: Point3D, boundingBox: BoundingBox)

    def buildNeighborsLookup[A : Order, B : Eq, D, T](things: Iterable[A], minDist: T)(key: A => Option[B])(using met: Metric[A, D, T]): Map[A, NonEmptySet[A]] = {
        val closePairs: List[(A, A)] = things.groupBy(key).filter(_._1.nonEmpty).values.toList.flatMap(_.toList.combinations(2).flatMap{
            case a1 :: a2 :: Nil => met.within(minDist)(a1, a2).option(a1 -> a2)
            case xs => throw new Exception(s"${xs.length} (not 2!) element(s) when taking pairwise combinations")
        })
        closePairs.foldLeft(Map.empty[A, NonEmptySet[A]]){ case (acc, (a1, a2)) => 
            val v1 = acc.get(a1).fold(NonEmptySet.one[A])(_.add)(a2)
            val v2 = acc.get(a2).fold(NonEmptySet.one[A])(_.add)(a1)
            acc ++ Map(a1 -> v1, a2 -> v2)
        }
    }
    
    def buildNeighboringRoisFinder(rois: List[(Roi, NonnegativeInt)], minDist: NonnegativeReal)(grouping: NEL[ProbeGroup])(
        using met: Metric[(Roi, NonnegativeInt), EuclideanDistance, EuclideanDistance.Threshold]): Either[String, Map[LineNumber, NonEmptySet[LineNumber]]] = {
        given ordByIndex: Order[Roi] = Order.by(_.index)
        val (groupIds, repeatedFrames) = NonnegativeInt.indexed(grouping.toList)
            .flatMap((g, i) => g.get.toList.map(_ -> i))
            .foldLeft(Map.empty[FrameIndex, NonnegativeInt] -> Map.empty[FrameIndex, Int]){ 
                case ((ids, repeats), (frame, gid)) =>
                    if ids `contains` frame
                    then (ids, repeats + (frame -> (repeats.getOrElse(frame, 0) + 1)))
                    else (ids + (frame -> gid), repeats)
            }
        repeatedFrames.isEmpty.either(
            s"${repeatedFrames.size} repeated frame(s): $repeatedFrames", 
            {
                val distanceTolerance = EuclideanDistance.Threshold(minDist)
                val neighborsByRoi = buildNeighborsLookup(rois, distanceTolerance)((r, _) => groupIds.get(r.time))
                neighborsByRoi.map{ case ((_, idx), indexedNeighbors) => idx -> indexedNeighbors.map(_._2) }
            }
            )
    }

    def rowToRoi(row: CsvRow): ErrMsgsOr[Roi] = {
        val indexNel = safeGetFromRow("", safeParseInt >>> RoiIndex.fromInt)(row)
        val posNel = safeGetFromRow("position", (_: String).asRight)(row)
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
                    sideX = Interval(xMin, xMax),
                    sideY = Interval(yMin, yMax),
                    sideZ = Interval(zMin, zMax)
                )
            )
        }
        (indexNel, posNel, timeNel, channelNel, centroidNel, bboxNel).mapN(Roi.apply).toEither
    }
    
    def rowToDriftRecord(row: CsvRow): ErrMsgsOr[DriftRecord] = {
        val posNel = safeGetFromRow("position", (_: String).asRight)(row)
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

    def safeParseIntLike = safeParseDouble >>> tryToInt

    /** Shift a point by a particular coarse drift correction. */
    def shiftPoint(drift: CoarseDrift)(point: Point3D) = (drift, point) match { 
        case (CoarseDrift(ZDir(delZ), YDir(delY), XDir(delX)), Point3D(XCoordinate(x), YCoordinate(y), ZCoordinate(z))) =>
            Point3D(XCoordinate(x - delX), YCoordinate(y - delY), ZCoordinate(z - delZ))
    }
    
    /****************************************************************************************************************
     * Ancillary types and functions
     ****************************************************************************************************************/

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

    final case class DimX(get: PosInt)
    final case class DimY(get: PosInt)
    final case class DimZ(get: PosInt)
    
    final case class Box(x: DimX, y: DimY, z: DimZ)
    
    final case class Interval[C <: Coordinate](lo: C, hi: C):
        require(lo < hi, s"Lower bound not less than upper bound: ($lo, $hi)")
    end Interval
    
    final case class BoundingBox(sideX: Interval[XCoordinate], sideY: Interval[YCoordinate], sideZ: Interval[ZCoordinate])

    def safeReadBool = (_: String) match {
        case "1" => true.some
        case "0" => false.some
        case _ => None
    }

    extension [A](rw: ReadWriter[List[A]])
        def toNel(context: String): ReadWriter[NEL[A]] = 
            rw.bimap(_.toList, _.toNel.getOrElse{ throw new Exception(s"$context: No elements to read as nonempty list!") })

    /** Push a value through to the right side of an {@code Either}, pairing with the wrapped value */
    extension [A, L, R](f: A => Either[L, R])
        def throughRight: A => Either[L, (A, R)] = a => f(a).map(a -> _)

end LabelAndFilterRois
