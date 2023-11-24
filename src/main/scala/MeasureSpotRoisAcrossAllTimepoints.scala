package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import upickle.default.*
import cats.syntax.apply.*
import cats.syntax.either.*
import cats.syntax.flatMap.*
import cats.syntax.option.*
import mouse.boolean.*
import scopt.OParser

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

    def workflow(spotsFile: os.Path, driftFile: os.Path, probeGroups: List[List[ProbeName]], minSpotSeparation: NonnegativeReal): Unit = {
        val delimiter = Delimiter.fromPathUnsafe(spotsFile)
        ???
    }

    type PosInt = PositiveInt

    final case class DimX(get: PosInt)
    final case class DimY(get: PosInt)
    final case class DimZ(get: PosInt)
    final case class Box(x: DimX, y: DimY, z: DimZ)

    final case class DimensionsSpecification(position: PositionIndex, time: FrameIndex, channel: NonnegativeInt, box: Box):
        final def x = box.x
        final def y = box.y
        final def z = box.z
    end DimensionsSpecification

    object DimensionsSpecification:
        given rwForDimensionSpecification: ReadWriter[DimensionsSpecification] = readwriter[ujson.Value].bimap(
            dimspec => ujson.Obj(
                "position" -> ujson.Num(dimspec.position.get), 
                "time" -> ujson.Num(dimspec.time.get), 
                "channel" -> ujson.Num(dimspec.channel), 
                "z" -> ujson.Num(dimspec.z.get), 
                "y" -> ujson.Num(dimspec.y.get), 
                "x" -> ujson.Num(dimspec.x.get),
            ), 
            json => {
                val posNel = fromJsonThruInt("position", PositionIndex.fromInt)(json)
                val timeNel = fromJsonThruInt("time", FrameIndex.fromInt)(json)
                val channelNel = fromJsonThruInt("channel", NonnegativeInt.either)(json)
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
