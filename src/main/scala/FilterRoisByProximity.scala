package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.effect.IO
import cats.syntax.all.*
import mouse.boolean.*
import scopt.OParser
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{ readCsvToCaseClasses, writeCaseClassesToCsv }

import at.ac.oeaw.imba.gerlich.looptrace.cli.scoptReaders.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*


/** Mutual exclusion of ROIs that are too close together */
object FilterRoisByProximity extends StrictLogging:
    val ProgramName = "FilterRoiByProximity"

    final case class CliConfig(
        inputFile: os.Path = null, // unconditionally required
        discardsOutputFile: os.Path = null, // unconditionally required
        usablesOutputFile: os.Path = null, // unconditionally required
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
                .text("Path to file from which to read data (merge-assessed ROI records)"),
            opt[os.Path]("discardsOutputFile")
                .required()
                .action((f, c) => c.copy(discardsOutputFile = f))
                .validate{ f => os.isDir(f.parent)
                    .either(f"Path to folder for merge contributors file isn't an extant folder: ${f.parent}", ())
                }
                .text("Path to the file to write merge contributor ROIs"),
            opt[os.Path]("usablesOutputFile")
                .required()
                .action((f, c) => c.copy(usablesOutputFile = f))
                .validate{ f => os.isDir(f.parent)
                    .either(f"Path to folder for merge results file isn't an extant folder: ${f.parent}", ())
                }
                .text("Path to the file to write merge contributor ROIs"),
            opt[Unit]("overwrite")
                .action((_, c) => c.copy(overwrite = true))
                .text("Allow overwriting output files."),
            checkConfig{ c => 
                val paths = List(c.inputFile, c.discardsOutputFile, c.usablesOutputFile)
                if paths.length === paths.toSet.size 
                then success
                else failure(s"Non-unique in/out paths for ROI merge: ${paths}")
            },
            checkConfig{ c => 
                if !c.overwrite && os.isFile(c.discardsOutputFile)
                then failure(s"Overwrite isn't authorised but output file exists: ${c.discardsOutputFile}")
                else success
            },
            checkConfig{ c => 
                if !c.overwrite && os.isFile(c.usablesOutputFile)
                then failure(s"Overwrite isn't authorised but output file exists: ${c.usablesOutputFile}")
                else success
            },
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => 
                // CLI parser gives error message.
                throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help")
            case Some(opts) => ???
        }
    }
end FilterRoisByProximity
