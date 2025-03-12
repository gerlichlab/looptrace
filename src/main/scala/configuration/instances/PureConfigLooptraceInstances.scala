package at.ac.oeaw.imba.gerlich.looptrace.configuration
package instances

import cats.syntax.all.*
import pureconfig.ConfigReader
import pureconfig.error.CannotConvert
import pureconfig.generic.semiauto.deriveReader
import squants.space.{ Length, LengthUnit }
import at.ac.oeaw.imba.gerlich.looptrace.space.*

/** PureConfig typeclasses instances for looptrace domain-specific types */
trait PureConfigLooptraceInstances:
    given (readLength: ConfigReader[Length]) => ConfigReader[LengthInNanometers] = 
        readLength.emap{ l => 
            LengthInNanometers.fromSquants(l).leftMap{ msg => 
                CannotConvert(value = l.toString, toType = "LengthInNanometers", because = msg) 
            }
        }

    given (readLengthInNanometers: ConfigReader[LengthInNanometers]) => ConfigReader[PixelDefinition] = 
        readLengthInNanometers.emap{ l => 
            PixelDefinition.tryToDefine(l).leftMap{ msg => 
                CannotConvert(value = l.toString, toType = "looptrace.space.PixelDefinition", because = msg) 
            }
        }

    given (ConfigReader[PixelDefinition]) => ConfigReader[Pixels3D] = 
        deriveReader[Pixels3D]
end PureConfigLooptraceInstances