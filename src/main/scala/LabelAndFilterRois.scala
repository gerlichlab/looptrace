package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ Failure, Success, Try }
import upickle.default.*
import cats.{ Alternative, Eq, Functor, Order }
import cats.data.{ NonEmptyList as NEL, NonEmptyMap, NonEmptySet, ValidatedNel }
import cats.data.Validated.Valid
import cats.syntax.all.*
import mouse.boolean.*

import scopt.OParser
import com.github.tototoshi.csv.*

import at.ac.oeaw.imba.gerlich.looptrace.space.{ Coordinate, EuclideanDistance, Metric, Point3D, XCoordinate, YCoordinate, ZCoordinate }
import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.{ fromJsonThruInt, safeExtract, readJsonFile }

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
        // TODO: bring back for inclusion of the rest of the spots table extraction here in this program.
        //imageSizesFile: os.Path = null, // unconditionally required
        probeGroupsFile: os.Path = null, // unconditionally required; specifies groups of probes which may not be too close
        minSpotSeparation: NonnegativeReal = NonnegativeReal(0), // unconditionally required
        unfilteredOutputFile: UnfilteredOutputFile = null, // unconditionally required
        filteredOutputFile: FilteredOutputFile = null, // unconditionally required
        extantOutputHandler: ExtantOutputHandler = null, // unconditionally required
        filterForNuclei: Boolean = false
        )

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import ScoptCliReaders.given
        import parserBuilder.*

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
            // TODO: bring back for inclusion of the rest of the spots table extraction here in this program.
            // opt[os.Path]("imageSizesFile")
            //     .required()
            //     .action((f, c) => c.copy(imageSizesFile = f))
            //     .validate(f => os.isFile(f).either(s"Alleged image sizes file isn't a file: $f", ()))
            //     .text("Path to file with dimensions (z, y, x) for each (position, time, channel)"),
            opt[os.Path]("probeGroupsFile")
                .required()
                .action((f, c) => c.copy(probeGroupsFile = f))
                .validate(f => os.isFile(f).either(f"Alleged probe groups file isn't a file: $f", ()))
                .text("Path to grouping of probes prohibited from being too close; should be simple list-of-lists in JSON"),
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
            // TODO: bring back for inclusion of the rest of the spots table extraction here in this program.
            // opt[Unit]("filterForNuclei")
            //     .action((_, c) => c.copy(filterForNuclei = true))
            //     .text("Use only spots that are in nuclei; only use this if also doing nuclei detection first.")
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => 
                
                // TODO: bring back for inclusion of the rest of the spots table extraction here in this program.
                // val dimsSpecs = {
                //     val context = "Parsing image dimensions specification file"
                //     println(s"$context: ${opts.imageSizesFile}")
                //     given rw: ReadWriter[NEL[DimensionsSpecification]] = summon[ReadWriter[List[DimensionsSpecification]]].toNel(context)
                //     readJsonFile[NEL[DimensionsSpecification]](opts.imageSizesFile)
                // }
                // workflow(opts.spotsFile, opts.driftFile, dimsSpecs, opts.probeGroupsFile, opts.minSpotSeparation, opts.filterForNuclei)
                workflow(
                    spotsFile = opts.spotsFile, 
                    driftFile = opts.driftFile, 
                    probeGroupsFile = opts.probeGroupsFile, 
                    minSpotSeparation = opts.minSpotSeparation, 
                    filterForNuclei = opts.filterForNuclei, 
                    unfilteredOutputFile = opts.unfilteredOutputFile,
                    filteredOutputFile = opts.filteredOutputFile, 
                    extantOutputHandler = opts.extantOutputHandler
                    )
        }
    }

    // TODO: bring back for inclusion of the rest of the spots table extraction here in this program.
    //def workflow(spotsFile: os.Path, driftFile: os.Path, dimsSpecs: NEL[DimensionsSpecification], probeGroups: List[ProbeGroup], minSpotSeparation: NonnegativeReal, filterForNuclei: Boolean): Unit = {
    def workflow(
        spotsFile: os.Path, 
        driftFile: os.Path, 
        probeGroupsFile: os.Path, 
        minSpotSeparation: NonnegativeReal, 
        filterForNuclei: Boolean, 
        unfilteredOutputFile: UnfilteredOutputFile, 
        filteredOutputFile: FilteredOutputFile, 
        extantOutputHandler: ExtantOutputHandler
        ): Unit = {
        
        // First, parse the probe groupings.
        val probeGroups = {
            println(s"Parsing probe groups file: ${probeGroupsFile}")
            readJsonFile[List[ProbeGroup]](probeGroupsFile)
        }

        /* Then, parse the ROI records from the (regional barcode) spots file. */
        val (roisHeader, rowRoiPairs): (List[String], List[((CsvRow, Roi), LineNumber)]) = {
            println(s"Reading ROIs file: $spotsFile")
            val reader = CSVReader.open(spotsFile.toIO)
            Try{ reader.allWithOrderedHeaders() } match {
                case Failure(exception) => 
                    reader.close()
                    throw exception
                case Success((head, spotRows)) => Alternative[List].separate(spotRows.map(rowToRoi.throughRight)) match {
                    case (Nil, rrPairs) => head -> NonnegativeInt.indexed(rrPairs)
                    case (errors@(h :: _), _) => throw new Exception(s"${errors.length} errors converting spot file (${spotsFile}) rows to ROIs! First one: $h")
                }
            }
        }
        
        /* Then, parse the drift correction records from the corresponding file. */
        val rowToDrift = rowToDriftRecord(filterForNuclei)
        println(s"Reading drift file: $driftFile")
        val drifts = withCsvData(driftFile){ (driftRows: Iterable[CsvRow]) => Alternative[List].separate(driftRows.toList.map(rowToDrift)) match {
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
            val wroteIt = writeAllCsv(unfilteredOutputFile, header, records, extantOutputHandler)
            println(s"${if wroteIt then "Wrote" else "Did not write"} unfiltered output file: $filteredOutputFile")
            header
        }
        println(s"Unfiltered header: $unfilteredHeader")

        val wroteIt = writeAllCsv(filteredOutputFile, roisHeader, roiRecordsLabeled.filter(_._2.isEmpty).map(_._1), extantOutputHandler)
        println(s"${if wroteIt then "Wrote" else "Did not write"} filtered output file: $filteredOutputFile")

        println("Done!")
    }

    /****************************************************************************************************************
     * Main types and business logic
     ****************************************************************************************************************/

    /** Specification of all dimensions of an image: where it's from (@code PositionIndex), along with (t, c, z, y, x) - like */
    // TODO: bring back for inclusion of the rest of the spots table extraction here in this program.
    // final case class DimensionsSpecification(position: String, time: FrameIndex, channel: Channel, box: Box):
    //     final def x = box.x
    //     final def y = box.y
    //     final def z = box.z
    // end DimensionsSpecification

    /** JSON representation of a dimensions specification */
    // TODO: bring back for inclusion of the rest of the spots table extraction here in this program.
    // object DimensionsSpecification:
    //     given rwForDimensionSpecification: ReadWriter[DimensionsSpecification] = readwriter[ujson.Value].bimap(
    //         dimspec => ujson.Obj(
    //             "position" -> ujson.Str(dimspec.position),
    //             "time" -> ujson.Num(dimspec.time.get), 
    //             "channel" -> ujson.Num(dimspec.channel.get), 
    //             "z" -> ujson.Num(dimspec.z.get), 
    //             "y" -> ujson.Num(dimspec.y.get), 
    //             "x" -> ujson.Num(dimspec.x.get),
    //         ), 
    //         json => {
    //             val posNel = safeExtract("position", identity)(json)
    //             val timeNel = fromJsonThruInt("time", FrameIndex.fromInt)(json)
    //             val channelNel = fromJsonThruInt("channel", Channel.fromInt)(json)
    //             val zNel = fromJsonThruInt("z", PositiveInt.either >> DimZ.apply)(json)
    //             val yNel = fromJsonThruInt("y", PositiveInt.either >> DimY.apply)(json)
    //             val xNel = fromJsonThruInt("x", PositiveInt.either >> DimX.apply)(json)
    //             val errsOrSpec = (posNel, timeNel, channelNel, zNel, yNel, xNel).mapN(
    //                 (pos, time, ch, z, y, x) => DimensionsSpecification(pos, time, ch, Box(x, y, z)))
    //             errsOrSpec.fold(errs => throw new Exception(s"${errs.size} errors parsing dims spec from JSON: ${errs}"), identity)
    //         }
    //     )
    // end DimensionsSpecification

    final case class DriftRecord(position: String, time: Timepoint, coarse: CoarseDrift, fine: FineDrift, inNucleus: Option[Boolean])

    final case class ProbeGroup(get: NEL[FrameIndex])
    object ProbeGroup:
        given rwForProbeGroup: ReadWriter[ProbeGroup] = readwriter[ujson.Value].bimap(
            group => ujson.Arr(group.get.toList.map(name => ujson.Num(name.get))*), 
            json => ProbeGroup(json.arr.toList.toNel
                .getOrElse{ throw new Exception("Empty collection can't parse as probe group!") }
                .map{ v => FrameIndex.unsafe(v.int) }
                )
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
        val indexNel = getFromRow("", safeParseInt >>> RoiIndex.fromInt)(row)
        val posNel = getFromRow("position", (_: String).asRight)(row)
        val timeNel = getFromRow("frame", safeParseInt >>> FrameIndex.fromInt)(row)
        val channelNel = getFromRow("ch", safeParseInt >>> Channel.fromInt)(row)
        val centroidNel = {
            val zNel = getFromRow("zc", safeParseDouble >> ZCoordinate.apply)(row)
            val yNel = getFromRow("yc", safeParseDouble >> YCoordinate.apply)(row)
            val xNel = getFromRow("xc", safeParseDouble >> XCoordinate.apply)(row)
            (zNel, yNel, xNel).mapN((z, y, x) => Point3D(x, y, z))
        }
        val bboxNel = {
            val zMinNel = getFromRow("z_min", safeParseDouble >> ZCoordinate.apply)(row)
            val zMaxNel = getFromRow("z_max", safeParseDouble >> ZCoordinate.apply)(row)
            val yMinNel = getFromRow("y_min", safeParseDouble >> YCoordinate.apply)(row)
            val yMaxNel = getFromRow("y_max", safeParseDouble >> YCoordinate.apply)(row)
            val xMinNel = getFromRow("x_min", safeParseDouble >> XCoordinate.apply)(row)
            val xMaxNel = getFromRow("x_max", safeParseDouble >> XCoordinate.apply)(row)
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
    
    def rowToDriftRecord(filterForNuclei: Boolean)(row: CsvRow): ErrMsgsOr[DriftRecord] = {
        val getNucNel: CsvRow => ValidatedNel[String, Option[Boolean]] = 
            if filterForNuclei
            then getFromRow("nuc_label", (s: String) => safeReadBool(s).toRight(s"Cannot parse text as boolean: $s").map(_.some))
            else Function.const{ Valid(Option.empty[Boolean]) }
        val posNel = getFromRow("position", (_: String).asRight)(row)
        val timeNel = getFromRow("frame", safeParseInt >>> FrameIndex.fromInt)(row)
        val coarseDriftNel = {
            val zNel = getFromRow("z_px_coarse", safeParseIntLike >> ZDir.apply)(row)
            val yNel = getFromRow("y_px_coarse", safeParseIntLike >> YDir.apply)(row)
            val xNel = getFromRow("x_px_coarse", safeParseIntLike >> XDir.apply)(row)
            (zNel, yNel, xNel).mapN(CoarseDrift.apply)
        }
        val fineDriftNel = {
            val zNel = getFromRow("z_px_fine", safeParseDouble >> ZDir.apply)(row)
            val yNel = getFromRow("x_px_fine", safeParseDouble >> YDir.apply)(row)
            val xNel = getFromRow("y_px_fine", safeParseDouble >> XDir.apply)(row)
            (zNel, yNel, xNel).mapN(FineDrift.apply)
        }
        val inNucNel = getNucNel(row)
        (posNel, timeNel, coarseDriftNel, fineDriftNel, inNucNel).mapN(DriftRecord.apply).toEither
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

    def getFromRow[A](key: String, lift: String => Either[String, A])(row: CsvRow) =
        (Try{ row(key) }.toEither.leftMap(_.getMessage) >>= lift).toValidatedNel

    def safeReadBool = (_: String) match {
        case "1" => true.some
        case "0" => false.some
        case _ => None
    }

    extension [A](rw: ReadWriter[List[A]])
        def toNel(context: String): ReadWriter[NEL[A]] = 
            rw.bimap(_.toList, _.toNel.getOrElse{ throw new Exception(s"$context: No elements to read as nonempty list!") })

    /** Add a continuation-like syntax for flatmapping over a function that can fail with another that can fail. */
    extension [A, L, B, R](f: A => Either[L, B])
        infix def >>>[L1 >: L](g: B => Either[L1, R]): A => Either[L1, R] = f(_: A).flatMap(g)
    
    /** Add a continuation-like syntax for flatmapping over a function that can fail with another that canNOT fail. */
    extension [A, L, B](f: A => Either[L, B])
        infix def >>[C](g: B => C): A => Either[L, C] = f(_: A).map(g)

    /** Push a value through to the right side of an {@code Either}, pairing with the wrapped value */
    extension [A, L, R](f: A => Either[L, R])
        def throughRight: A => Either[L, (A, R)] = a => f(a).map(a -> _)

end LabelAndFilterRois
