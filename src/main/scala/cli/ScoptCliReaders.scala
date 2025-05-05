package at.ac.oeaw.imba.gerlich.looptrace.cli

import scala.annotation.targetName
import cats.syntax.all.*
import io.github.iltotore.iron.:|
import io.github.iltotore.iron.constraint.numeric.{Negative, Positive}
import io.github.iltotore.iron.constraint.any.Not
import scopt.Read
import squants.space.Length

import at.ac.oeaw.imba.gerlich.gerlib.geometry.{Distance, EuclideanDistance}
import at.ac.oeaw.imba.gerlich.gerlib.imaging.{ImagingTimepoint, Pixels3D}
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.refinement.IllegalRefinement
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration

object scoptReaders extends ScoptCliReaders

/** Allow custom types as CLI parameters. */
trait ScoptCliReaders:
  given (fileRead: Read[java.io.File]) => Read[os.Path] =
    fileRead.map(os.Path.apply)

  @targetName("readNonnegativeInt")
  given (intRead: Read[Int]) => Read[Int :| Not[Negative]] =
    intRead.map(z =>
      NonnegativeInt.option(z).getOrElse {
        throw IllegalRefinement(z, "Cannot refine as nonnegative")
      }
    )

  @targetName("readNonnegativeDouble")
  given (numRead: Read[Double]) => Read[Double :| Not[Negative]] =
    numRead.map(x =>
      NonnegativeReal.option(x).getOrElse {
        throw IllegalRefinement(x, "Cannot refine as nonnegative")
      }
    )

  @targetName("readPositiveInt")
  given (intRead: Read[Int]) => Read[Int :| Positive] =
    intRead.map(z =>
      PositiveInt.option(z).getOrElse {
        throw IllegalRefinement(z, "Cannot refine as positive")
      }
    )

  @targetName("readPositiveDouble")
  given (numRead: Read[Double]) => Read[Double :| Positive] =
    numRead.map(x =>
      PositiveReal.option(x).getOrElse {
        throw IllegalRefinement(x, "Cannot refine as positive")
      }
    )

  given Read[Length]:
    override def reads: String => Length = s =>
      Length(s).fold(throw _, identity)

  given (lenRead: Read[Length]) => Read[Distance] = lenRead.map { l =>
    Distance
      .option(l)
      .getOrElse(throw IllegalRefinement(l, "Cannot refine length as distance"))
  }

  def summonEuclideanDistanceReader(using
      readDist: Read[Distance]
  ): Read[EuclideanDistance] =
    readDist.map(EuclideanDistance.apply)

  given (readNN: Read[Int :| Not[Negative]]) => Read[ImagingTimepoint] =
    readNN.map(ImagingTimepoint.apply)

  /** Parse content of JSON file path to imaging rounds configuration instance.
    */
  given readForImagingRoundsConfiguration
      : scopt.Read[ImagingRoundsConfiguration] = scopt.Read.reads { file =>
    ImagingRoundsConfiguration.fromJsonFile(os.Path(file)) match {
      case Left(messages) =>
        throw new IllegalArgumentException(
          s"Cannot read file ($file) as imaging round configuration! Error(s): ${messages.mkString_("; ")}"
        )
      case Right(conf) => conf
    }
  }

  // Use the pureconfig.ConfigReader instance to parse a CLI argument specifying pixel definitions
  given scopt.Read[Pixels3D] =
    import pureconfig.*
    import at.ac.oeaw.imba.gerlich.looptrace.configuration.instances.all.given
    scopt.Read.reads { s =>
      ConfigSource
        .string(s)
        .load[Pixels3D]
        .leftMap(_.prettyPrint)
        .leftMap(msg =>
          new IllegalArgumentException(s"Cannot decode pixel scaling: $msg")
        )
        .fold(throw _, identity)
    }
end ScoptCliReaders
