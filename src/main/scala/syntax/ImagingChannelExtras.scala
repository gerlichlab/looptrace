package at.ac.oeaw.imba.gerlich.looptrace
package syntax

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingChannel
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt

object ImagingChannelExtras:
    extension (IC: ImagingChannel.type)
        def fromInt: Int => Either[String, ImagingChannel] = NonnegativeInt.either >> ImagingChannel.apply
        def unsafe: Int => ImagingChannel = NonnegativeInt.unsafe `andThen` ImagingChannel.apply
