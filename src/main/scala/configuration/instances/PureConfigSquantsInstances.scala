package at.ac.oeaw.imba.gerlich.looptrace.configuration
package instances

import scala.reflect.ClassTag
import scala.util.Try
import cats.syntax.all.*
import pureconfig.ConfigReader
import pureconfig.error.{ CannotConvert, FailureReason }
import squants.space.Length
import at.ac.oeaw.imba.gerlich.looptrace.space.LengthInNanometers

/** PureConfig t ypeclass instances for squants */
trait PureConfigSquantsInstances:
    given configReaderForLength(using readString: ConfigReader[String]): ConfigReader[Length] = 
        readString.emap: s => 
            Length.parseString(s)
                .toEither
                .leftMap{ e => CannotConvert(value = s, toType = "squants.space.Length", because = e.getMessage) }
end PureConfigSquantsInstances
