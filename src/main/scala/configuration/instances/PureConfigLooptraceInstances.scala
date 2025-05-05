package at.ac.oeaw.imba.gerlich.looptrace.configuration
package instances

import pureconfig.ConfigReader
import pureconfig.generic.semiauto.deriveReader
import squants.space.{Length, LengthUnit}

import at.ac.oeaw.imba.gerlich.gerlib.imaging.{Pixels3D, PixelDefinition}
import at.ac.oeaw.imba.gerlich.looptrace.space.*

/** PureConfig typeclasses instances for looptrace domain-specific types */
trait PureConfigLooptraceInstances:
  given (ConfigReader[PixelDefinition]) => ConfigReader[Pixels3D] =
    deriveReader[Pixels3D]

  given ConfigReader[PixelDefinition] = ???
end PureConfigLooptraceInstances
