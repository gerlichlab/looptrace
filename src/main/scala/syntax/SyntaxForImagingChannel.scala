package at.ac.oeaw.imba.gerlich.looptrace
package syntax

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingChannel
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt
import at.ac.oeaw.imba.gerlich.gerlib.refinement.IllegalRefinement
import at.ac.oeaw.imba.gerlich.looptrace.syntax.function.*

trait SyntaxForImagingChannel:
  extension (IC: ImagingChannel.type)
    def fromInt: Int => Either[String, ImagingChannel] =
      NonnegativeInt.either >> ImagingChannel.apply
    def unsafe: Int => ImagingChannel = z =>
      NonnegativeInt
        .option(z)
        .fold(
          throw IllegalRefinement(
            z,
            s"Alleged value for imaging channel ($z) isn't nonnegative"
          )
        )(ImagingChannel.apply)
