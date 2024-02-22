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
object SummariseImagingRoundsConfiguration:
    val ProgramName = "SummariseImagingRoundsConfiguration"
    
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
                val config = ImagingRoundsConfiguration.unsafeFromJsonFile(opts.configFile)
                val exclusions = config.tracingExclusions.map(_.get)
                println(s"${exclusions.size} exclusion(s) from tracing: ${exclusions.toList.sorted.map(_.show).mkString(", ")}")
                println(s"${config.numberOfRounds} round(s) in total (listed below)")
                config.allRounds.map(r => s"${r.time.show}: ${r.name.show}").toList.foreach(println)
                val (groupingName, maybeGroups) = config.regionGrouping match {
                    case ImagingRoundsConfiguration.RegionGrouping.Trivial => "Trivial" -> None
                    case grouping: ImagingRoundsConfiguration.RegionGrouping.Permissive => "Permissive" -> grouping.groups.some
                    case grouping: ImagingRoundsConfiguration.RegionGrouping.Prohibitive => "Prohibitive" -> grouping.groups.some
                }
                println(s"$groupingName regional grouping")
                maybeGroups.fold(()){ groups => 
                    groups.zipWithIndex.toList.foreach{ (g, i) => 
                        println(s"$i: ${g.toList.sorted.mkString(", ")}")
                    }
                }
        }
    }
end SummariseImagingRoundsConfiguration
