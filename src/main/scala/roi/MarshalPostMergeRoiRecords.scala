package at.ac.oeaw.imba.gerlich.looptrace
package roi

import cats.*
import cats.effect.IO
import cats.syntax.all.*
import fs2.data.csv.*
import mouse.boolean.*
import scopt.OParser
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{
    getCsvRowDecoderForProduct2, 
    readCsvToCaseClasses, 
    writeCaseClassesToCsv,
}
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.cli.scoptReaders.given
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.IndexedDetectedSpot
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergedRoiRecord
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext

/**
 * Collect the singleton ROIs and the merge result ROIs into one file.
 * 
 * This is useful such that the proximity filtration program can consider 
 * all ROIs jointly, as is certainly necessary, without regard for whether 
 * an ROI comes from a merge or is as was initially detected. 
 *
 * Otherwise, a merged ROI too close to an unmerged ROI would not properly 
 * lead to a mutual exclusion / discard event.
 */
object MarshalPostMergeRoiRecords extends StrictLogging:
    val ProgramName = "MarshalPostMergeRoiRecords"

    final case class CliConfig(
        unmergedInputFile: os.Path = null, // unconditionally required
        mergedInputFile: os.Path = null, // unconditionally required
        outputFile: os.Path = null, // unconditionally required
        overwrite: Boolean = false,
    )

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import parserBuilder.*
        
        given Eq[os.Path] = Eq.by(_.toString)

        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, BuildInfo.version), 
            opt[os.Path]("unmergedInputFile")
                .required()
                .action((f, c) => c.copy(unmergedInputFile = f))
                .validate{ f => os.isFile(f)
                    .either(s"Alleged unmerged input path isn't extant file: $f", ())
                }
                .text("Path to the unmerged ROIs file"),
            opt[os.Path]("mergedInputFile")
                .required()
                .action((f, c) => c.copy(mergedInputFile = f))
                .validate{ f => os.isFile(f)
                    .either(s"Alleged merged input path isn't extant file: $f", ())
                }
                .text("Path to the merged ROIs file"),
            opt[os.Path]('O', "outputFile")
                .required()
                .action((f, c) => c.copy(outputFile = f))
                .text("Path to output file to write"),
            opt[Unit]("overwrite")
                .action((_, c) => c.copy(overwrite = false))
                .text("Allow overwriting of a pre-existing output file."), 
            checkConfig{ c => 
                if c.unmergedInputFile === c.mergedInputFile 
                then failure(s"Unmerged and merged input files are the same: ${c.mergedInputFile}")
                else success
            },
            checkConfig{ c => 
                if c.unmergedInputFile === c.outputFile 
                then failure(s"Unmerged input file and output file are the same: ${c.unmergedInputFile}")
                else success
            },
            checkConfig{ c => 
                if c.mergedInputFile === c.outputFile 
                then failure(s"Merged input file and output file are the same: ${c.mergedInputFile}")
                else success
            },
            checkConfig{ c => 
                if c.overwrite || !os.exists(c.outputFile) then success
                else failure(s"Overwrite isn't authorised but output file already exists: ${c.outputFile}")
            }
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => 
                // CLI parser gives error message.
                throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help")
            case Some(opts) => 
                import cats.effect.unsafe.implicits.global // needed for cats.effect.IORuntime
                import fs2.data.text.utf8.* // for CharLikeChunks typeclass instances

                /* Build up the program. */
                logger.info(s"Will read unmerged records from ${opts.unmergedInputFile}")
                val readUnmerged: IO[List[IndexedDetectedSpot]] = 
                    readCsvToCaseClasses[IndexedDetectedSpot](opts.unmergedInputFile)
                logger.info(s"Will read merged records from ${opts.mergedInputFile}")
                val readMerged: IO[List[MergedRoiRecord]] = 
                    readCsvToCaseClasses[MergedRoiRecord](opts.mergedInputFile)
                
                val write: List[IndexedDetectedSpot | MergedRoiRecord] => IO[Unit] = 
                    rois => fs2.Stream.emits(rois)
                        .through(writeCaseClassesToCsv(opts.outputFile))
                        .compile
                        .drain
                val prog: IO[Unit] = Applicative[IO].map2(readUnmerged, readMerged)(_ ::: _)
                
                /* Run the program. */
                logger.info(s"Will write output file: ${opts.outputFile}")
                prog.unsafeRunSync()
                logger.info("Done!")
        }
    }

    given CsvRowEncoder[IndexedDetectedSpot | MergedRoiRecord, String] with
        override def apply(elem: IndexedDetectedSpot | MergedRoiRecord): RowF[Some, String] = 
            elem match {
                case (idx, roi) => 
                    val idxRow = summon[CsvRowEncoder[RoiIndex, String]](idx)
                    val ctxRow = summon[CsvRowEncoder[ImagingContext, String]](roi.context)
                    val centroidRow = summon[CsvRowEncoder[Centroid[Double], String]](roi.centroid)
                    idxRow |+| ctxRow |+| centroidRow |+| ColumnNames.MergeRoisColumnName.write(Set())
                case merged: MergedRoiRecord => summon[CsvRowEncoder[MergedRoiRecord, String]](merged)
            }

    given CsvRowDecoder[IndexedDetectedSpot, String] = 
        // Parse the index directly, and decode the ROI components to build it.
        getCsvRowDecoderForProduct2{ (idx: RoiIndex, roi: DetectedSpotRoi) => idx -> roi }
end MarshalPostMergeRoiRecords
