package at.ac.oeaw.imba.gerlich.looptrace.cli

import scopt.Read
import cats.syntax.all.*

import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration

/** Allow custom types as CLI parameters. */
object ScoptCliReaders:
    given pathRead(using fileRead: Read[java.io.File]): Read[os.Path] = fileRead.map(os.Path.apply)
    given nonNegIntRead(using intRead: Read[Int]): Read[NonnegativeInt] = intRead.map(NonnegativeInt.unsafe)
    given nonNegRealRead(using numRead: Read[Double]): Read[NonnegativeReal] = numRead.map(NonnegativeReal.unsafe)
    given posIntRead(using intRead: Read[Int]): Read[PositiveInt] = intRead.map(PositiveInt.unsafe)
    given posRealRead(using numRead: Read[Double]): Read[PositiveReal] = numRead.map(PositiveReal.unsafe)
    /** Parse content of JSON file path to imaging rounds configuration instance. */
    given readForImagingRoundsConfiguration: scopt.Read[ImagingRoundsConfiguration] = scopt.Read.reads{ file => 
        ImagingRoundsConfiguration.fromJsonFile(os.Path(file)) match {
            case Left(messages) => throw new IllegalArgumentException(
                s"Cannot read file ($file) as imaging round configuration! Error(s): ${messages.mkString_("; ")}"
                )
            case Right(conf) => conf
        }
    }
end ScoptCliReaders
