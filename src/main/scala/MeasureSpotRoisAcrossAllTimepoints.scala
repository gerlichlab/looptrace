package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import upickle.default.*
import cats.{ Alternative, Eq, Order }
import cats.data.{ NonEmptyList as NEL, NonEmptySet, ValidatedNel }
import cats.syntax.all.*
import mouse.boolean.*
import scopt.OParser
import at.ac.oeaw.imba.gerlich.looptrace.space.{ Coordinate, XCoordinate, YCoordinate, ZCoordinate }
import javax.sound.sampled.Line

/**
 * Measure data across all timepoints in the regions identified during spot detection.
 * 
 * Optionally, also filter out spots that are too close together (e.g., because disambiguation 
 * of spots from indiviudal FISH probes in each region would be impossible in a multiplexed 
 * experiment).
 */
object MeasureSpotRoisAcrossAllTimepoints:
    val ProgramName = "MeasureSpotRoisAcrossAllTimepoints"

    type Timepoint = FrameIndex

    case class CliConfig(
        spotsFile: os.Path = null, // unconditionally required
        driftFile: os.Path = null, // unconditionally required
        imageSizesFile: os.Path = null, // unconditionally required
        probeGroupsFile: Option[os.Path] = None, // triggers filtering out of too-proximal spots
        minSpotSeparation: NonnegativeReal = NonnegativeReal(0), // required iff probe groupings file is provided
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
            opt[os.Path]("imageSizesFile")
                .required()
                .action((f, c) => c.copy(imageSizesFile = f))
                .validate(f => os.isFile(f).either(s"Alleged image sizes file isn't a file: $f", ()))
                .text("Path to file with dimensions (z, y, x) for each (position, time, channel)"),
            opt[os.Path]("probeGroupsFile")
                .action((f, c) => c.copy(probeGroupsFile = f.some))
                .validate(f => os.isFile(f).either(f"Alleged probe groups file isn't a file: $f", ()))
                .text("Path to grouping of probes prohibited from being too close; should be simple list-of-lists in JSON")
                .children(
                    opt[NonnegativeReal]("minSpotSeparation")
                        .action((px, c) => c.copy(minSpotSeparation = px))
                        .text("Minimum number of pixels required between centroids of a pair of spots; discard otherwise")
                )
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => 
                val outfolder = opts.spotsFile.parent
                ???
        }
    }

    def workflow(spotsFile: os.Path, driftFile: os.Path, probeGroups: List[NEL[FrameIndex]], minSpotSeparation: NonnegativeReal): Unit = {
        withCsvData(spotsFile){ (spotRows: Iterable[CsvRow]) =>
            val rois = NonnegativeInt.indexed(spotRows.toList)
            val lookupNeighbors: LineNumber => Option[NonEmptySet[LineNumber]] = probeGroups.toNel match {
                case None => { (_: LineNumber) => None }
                case Some(groups) => buildNeighborsLookup(rois, minSpotSeparation)(groups.map(ProbeGroup.apply))
            }
        }
    }

    trait Metric[A, D : Order]:
        def distanceBetween(a1: A, a2: A): D
        final def within(d: D) = distanceBetween(_: A, _: A) < d
    end Metric

    given roiMetricEuclidean: Metric[Roi, Double] with
        def distanceBetween(a1: Roi, a2: Roi): Double = ???

    def buildNeighborsLookup[A : Order, B : Eq, D : Order](things: Iterable[A], minDist: D)(key: A => Option[B])(using met: Metric[A, D]): Map[A, NonEmptySet[A]] = {
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
    
    def buildNeighborsLookup(rois: Iterable[Roi], minDist: NonnegativeReal)(grouping: NEL[ProbeGroup]): LineNumber => Option[NonEmptySet[LineNumber]] = {
        given ordByIndex: Order[Roi] = Order.by(_.index)
        val (groupIds, repeatedFrames) = NonnegativeInt.indexed(grouping.toList)
            .flatMap((g, i) => g.get.toList.map(_ -> i))
            .foldLeft(Map.empty[FrameIndex, NonnegativeInt] -> Map.empty[FrameIndex, Int]){ 
                case ((ids, repeats), (frame, gid)) =>
                    if ids `contains` frame
                    then (ids, repeats + (frame -> (repeats.getOrElse(frame, 0) + 1)))
                    else (ids + (frame -> gid), repeats)
            }
        if (repeatedFrames.nonEmpty) { throw new Exception(s"${repeatedFrames.size} repeated frame(s): $repeatedFrames") }
        val neighborsByRoi = buildNeighborsLookup(NonnegativeInt.indexed(rois.toList), minDist){ (roi: Roi, idx: LineNumber) => groupIds.get(roi.time) }
        val maybeNeighborsLineNumsByLineNum: Map[LineNumber, NonEmptySet[LineNumber]] = 
            neighborsByRoi.map{ case ((_, idx), indexedNeighbors) => idx -> indexedNeighbors.map(_._2) }
        maybeNeighborsLineNumsByLineNum.get
    }
    def getFromRow[A](key: String, lift: String => Either[String, A])(row: CsvRow) =
        (Try{ row(key) }.toEither.leftMap(_.getMessage) >>= lift).toValidatedNel

    def rowToRoi(positions: Map[String, Int])(row: CsvRow): ErrMsgsOr[Roi] = {
        val getPosIdx = (p: String) => positions.get(p).toRight(s"Unknown position: $p") >>= PositionIndex.fromInt
        val indexNel = getFromRow("", safeParseInt >>> RoiIndex.fromInt)(row)
        val posNel = getFromRow("position", getPosIdx)(row)
        val timeNel = getFromRow("frame", safeParseInt >>> FrameIndex.fromInt)(row)
        val channelNel = getFromRow("channel", safeParseInt >>> Channel.fromInt)(row)
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
        (indexNel, posNel, timeNel, channelNel, bboxNel).mapN(Roi.apply).toEither
    }

    extension [A, L, B, R](f: A => Either[L, B])
        infix def >>>[L1 >: L](g: B => Either[L1, R]): A => Either[L1, R] = f(_: A).flatMap(g)
    
    extension [A, L, B](f: A => Either[L, B])
        infix def >>[C](g: B => C): A => Either[L, C] = f(_: A).map(g)
    
    case class Roi(index: RoiIndex, position: PositionIndex, time: FrameIndex, channel: Channel, boundingBox: BoundingBox)

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

    type LineNumber = NonnegativeInt
    type PosInt = PositiveInt

    final case class DimX(get: PosInt)
    final case class DimY(get: PosInt)
    final case class DimZ(get: PosInt)
    
    final case class Box(x: DimX, y: DimY, z: DimZ)
    
    final case class Interval[C <: Coordinate](lo: C, hi: C):
        require(lo < hi, s"Lower bound not less than upper bound: ($lo, $hi)")
    end Interval
    
    final case class BoundingBox(sideX: Interval[XCoordinate], sideY: Interval[YCoordinate], sideZ: Interval[ZCoordinate])

    final case class DimensionsSpecification(position: PositionIndex, time: FrameIndex, channel: Channel, box: Box):
        final def x = box.x
        final def y = box.y
        final def z = box.z
    end DimensionsSpecification

    object DimensionsSpecification:
        given rwForDimensionSpecification: ReadWriter[DimensionsSpecification] = readwriter[ujson.Value].bimap(
            dimspec => ujson.Obj(
                "position" -> ujson.Num(dimspec.position.get), 
                "time" -> ujson.Num(dimspec.time.get), 
                "channel" -> ujson.Num(dimspec.channel.get), 
                "z" -> ujson.Num(dimspec.z.get), 
                "y" -> ujson.Num(dimspec.y.get), 
                "x" -> ujson.Num(dimspec.x.get),
            ), 
            json => {
                val posNel = fromJsonThruInt("position", PositionIndex.fromInt)(json)
                val timeNel = fromJsonThruInt("time", FrameIndex.fromInt)(json)
                val channelNel = fromJsonThruInt("channel", Channel.fromInt)(json)
                val zNel = fromJsonThruInt("z", PositiveInt.either.andThen(_.map(DimZ.apply)))(json)
                val yNel = fromJsonThruInt("y", PositiveInt.either.andThen(_.map(DimY.apply)))(json)
                val xNel = fromJsonThruInt("x", PositiveInt.either.andThen(_.map(DimX.apply)))(json)
                val errsOrSpec = (posNel, timeNel, channelNel, zNel, yNel, xNel).mapN(
                    (pos, time, ch, z, y, x) => DimensionsSpecification(pos, time, ch, Box(x, y, z)))
                errsOrSpec.fold(errs => throw new Exception(s"${errs.size} errors parsing dims spec from JSON: ${errs}"), identity)
            }
        )
    end DimensionsSpecification

    def fromJsonThruInt[A](key: String, lift: Int => Either[String, A]) = 
        (json: ujson.Value) => (Try{ json(key).int }.toEither.leftMap(_.getMessage) >>= lift).toValidatedNel

end MeasureSpotRoisAcrossAllTimepoints
