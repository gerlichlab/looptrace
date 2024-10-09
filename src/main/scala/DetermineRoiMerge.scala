package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.effect.IO
import cats.syntax.all.*
import fs2.data.csv.*
import mouse.boolean.*
import scopt.OParser
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.geometry.{ DistanceThreshold, PiecewiseDistance }
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeReal
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{ 
    getCsvRowDecoderForTuple2, 
    readCsvToCaseClasses, 
    writeCaseClassesToCsv, 
}
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given

import at.ac.oeaw.imba.gerlich.looptrace.cli.scoptReaders.given
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.roi.{ DetectedSpotRoi, MergerAssessedRoi }
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.{ IndexedDetectedSpot, assessForMerge }
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Consider the collection of detected spot ROIs for which ones to merge. */
object DetermineRoiMerge extends StrictLogging:
    val ProgramName = "DetermineRoiMerge"

    final case class CliConfig(
        inputFile: os.Path = null, // unconditionally required
        outputFile: os.Path = null, // unconditionally required
        distanceThreshold: DistanceThreshold = null, // unconditionally required,
        overwrite: Boolean = false,
    )

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import parserBuilder.*

        given Eq[os.Path] = Eq.by(_.toString)

        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, BuildInfo.version), 
            opt[os.Path]('I', "inputFile")
                .required()
                .action((f, c) => c.copy(inputFile = f))
                .validate(f => os.isFile(f).either(s"Alleged input path isn't an extant file: $f", ()))
                .text("Path to file from which to read data (detected ROI records)"),
            opt[os.Path]('O', "outputFile")
                .required()
                .action((f, c) => c.copy(outputFile = f))
                .validate{ f => os.isDir(f.parent)
                    .either(f"Path to folder for ROI merge assessment output file isn't an extant folder: ${f.parent}", ())
                }
                .text("Path to the output file to write"),
            opt[NonnegativeReal]('D', "distanceThreshold")
                .required()
                .action((d, c) => c.copy(distanceThreshold = PiecewiseDistance.ConjunctiveThreshold(d)))
                .text("Distance of centroid separation, beneath which ROIs will be merged"),
            opt[Unit]("overwrite")
                .action((_, c) => c.copy(overwrite = true))
                .text("Allow overwriting output file."),
            checkConfig{ c => 
                if c.inputFile === c.outputFile 
                then failure(s"Input and output file are the same: ${c.inputFile}")
                else success
            },
            checkConfig{ c =>
                if os.isFile(c.outputFile) && !c.overwrite 
                then failure(s"Overwrite isn't authorised but output file exists: ${c.outputFile}")
                else success
            }
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => 
                // CLI parser gives error message.
                throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help")
            case Some(opts) => 
                import cats.effect.unsafe.implicits.global // needed for cats.effect.IORuntime
                import fs2.data.text.utf8.* // for CharLikeChunks typeclass instances

                given CsvRowDecoder[IndexedDetectedSpot, String] = 
                    getCsvRowDecoderForTuple2[RoiIndex, DetectedSpotRoi, String]

                /* Build up the program. */
                val read: os.Path => IO[List[IndexedDetectedSpot]] = 
                    readCsvToCaseClasses[IndexedDetectedSpot] // TODO: adapt the Decoder to grab the index.
                val write: os.Path => (List[MergerAssessedRoi] => IO[Unit]) = 
                    outfile => {
                        fs2.Stream.emits(_)
                            .through(writeCaseClassesToCsv(outfile))
                            .compile
                            .drain
                    }
                val prog: IO[Unit] = read(opts.inputFile)
                    .map(assessForMerge(opts.distanceThreshold))
                    .flatMap(_.fold(
                        errors => 
                            throw new Exception(
                                s"${errors.length} error(s) determining spots' merger. First one: ${errors.head}"
                            ), 
                        write(opts.outputFile)
                    ))
                
                /* Run the program. */
                logger.info(s"Reading from: ${opts.inputFile}")
                prog.unsafeRunSync()
                logger.info("Done!")
        }
    }
end DetermineRoiMerge
