package at.ac.oeaw.imba.gerlich.looptrace.cli

import scopt.Read
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanDistance
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration
import at.ac.oeaw.imba.gerlich.looptrace.space.Pixels3D

object scoptReaders extends ScoptCliReaders

/** Allow custom types as CLI parameters. */
trait ScoptCliReaders:
    given (fileRead: Read[java.io.File]) => Read[os.Path] = fileRead.map(os.Path.apply)
    
    given (intRead: Read[Int]) => Read[NonnegativeInt] = intRead.map(NonnegativeInt.unsafe)
    
    given (numRead: Read[Double]) => Read[NonnegativeReal] = numRead.map(NonnegativeReal.unsafe)
    
    given (intRead: Read[Int]) => Read[PositiveInt] = intRead.map(PositiveInt.unsafe)
    
    given (numRead: Read[Double]) => Read[PositiveReal] = numRead.map(PositiveReal.unsafe)

    given (numRead: Read[Double]) => Read[EuclideanDistance.Threshold] = 
        numRead.map(NonnegativeReal.unsafe `andThen` EuclideanDistance.Threshold.apply)

    given (intRead: Read[Int]) => Read[ImagingTimepoint] = 
        intRead.map(NonnegativeInt.unsafe `andThen` ImagingTimepoint.apply)

    /** Parse content of JSON file path to imaging rounds configuration instance. */
    given readForImagingRoundsConfiguration: scopt.Read[ImagingRoundsConfiguration] = scopt.Read.reads{ file => 
        ImagingRoundsConfiguration.fromJsonFile(os.Path(file)) match {
            case Left(messages) => throw new IllegalArgumentException(
                s"Cannot read file ($file) as imaging round configuration! Error(s): ${messages.mkString_("; ")}"
                )
            case Right(conf) => conf
        }
    }

    // Use the pureconfig.ConfigReader instance to parse a CLI argument specifying pixel definitions
    given scopt.Read[Pixels3D] = 
        import pureconfig.*
        import at.ac.oeaw.imba.gerlich.looptrace.configuration.instances.all.given
        scopt.Read.reads{ s => 
            ConfigSource.string(s)
                .load[Pixels3D]
                .leftMap(_.prettyPrint)
                .leftMap(msg => new IllegalArgumentException(s"Cannot decode pixel scaling: $msg"))
                .fold(throw _, identity)
        }
end ScoptCliReaders
