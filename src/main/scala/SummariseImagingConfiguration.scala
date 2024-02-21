package at.ac.oeaw.imba.gerlich.looptrace

import upickle.default.*
import cats.*
import cats.data.*
import cats.syntax.all.*
import mouse.boolean.*
import scopt.OParser

/**
 * Summarise an imaging round configuration with command-line printing.
 * 
 * @author Vince Reuter
 */
object SummariseImagingRoundConfiguration:
    val ProgramName = "SummariseImagingRoundConfiguration"
    
    final case class CliConfig(configFile: os.Path = null)
    val parserBuilder = OParser.builder[CliConfig]
    
    def main(args: Array[String]): Unit = {
        import ScoptCliReaders.given
        import parserBuilder.*
        
        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, VersionName), 
            opt[os.Path]('C', "config")
                .required()
                .action((f, c) => c.copy(configFile = f))
                .validate(f => os.isFile(f).either(s"Alleged config file isn't extant file: $f", ()))
                .text("Path to configuration file to summarise")
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => 
                println(s"Reading config file: ${opts.configFile}")
                ImagingRoundConfiguration.fromJsonFile(opts.configFile) match {
                    case Left(messages) => throw new Exception(s"Error(s): ${messages.mkString_(", ")}")
                    case Right(config) => 
                        val exclusions = config.tracingExclusions.map(_.get)
                        println(s"${exclusions.size} exclusion(s) from tracing: ${exclusions.toList.sorted.map(_.show).mkString(", ")}")
                        println(s"${config.numberOfRounds} round(s) in total (listed below)")
                        config.sequenceOfRounds.rounds.map(r => s"${r.time.show}: ${r.name}").toList.foreach(println)
                        val (groupingName, maybeGroups) = config.regionalGrouping match {
                            case ImagingRoundConfiguration.RegionalGrouping.Trivial => "Trivial" -> None
                            case grouping: ImagingRoundConfiguration.RegionalGrouping.Permissive => "Permissive" -> grouping.groups.some
                            case grouping: ImagingRoundConfiguration.RegionalGrouping.Prohibitive => "Prohibitive" -> grouping.groups.some
                        }
                        println(s"$groupingName regional grouping")
                        maybeGroups.fold(()){ groups => 
                            groups.zipWithIndex.toList.foreach{ (g, i) => 
                                println(s"$i: ${g.toList.sorted.mkString(", ")}")
                            }
                        }
                }
        }
    }
end SummariseImagingRoundConfiguration
