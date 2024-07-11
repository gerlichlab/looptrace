package at.ac.oeaw.imba.gerlich.looptrace

import cats.syntax.all.*
import mouse.boolean.*
import com.typesafe.scalalogging.StrictLogging
import scopt.*

import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo

/**
  * Validate the imaging rounds configuration file.
  * 
  * @author Vince Reuter
  */
object ValidateImagingRoundsConfig extends StrictLogging:
    val ProgramName = "ValidateImagingRoundsConfig"
    
    final case class CliConfig(
        configFile: os.Path = null, // unconditionally required
    )
    
    val parserBuilder = OParser.builder[CliConfig]
    
    def main(args: Array[String]): Unit = {
        import parserBuilder.*
        import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders.given
        
        val parser = OParser.sequence(
            programName(ProgramName),
            // TODO: better naming and versioning
            head(ProgramName, BuildInfo.version),
            arg[os.Path]("<file>")
                .action((f, c) => c.copy(configFile = f))
                .validate(f => os.isFile(f).either(s"Alleged config to validate isn't an extant file: $f", ()))
                .text("Path to the config file to validate"),
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => ImagingRoundsConfiguration.fromJsonFile(opts.configFile) match {
                case Right(value) => logger.info(s"Successfully validated config ${opts.configFile}")
                case Left(errors) => 
                    logger.error(s"Validation failed for config ${opts.configFile}! Reasons will be listed below.")
                    errors.toList.foreach(logger.error)
                    throw new ImagingRoundsConfiguration.BuildError.FromJsonFile(messages = errors, file = opts.configFile)
            }
        }
    }
end ValidateImagingRoundsConfig
