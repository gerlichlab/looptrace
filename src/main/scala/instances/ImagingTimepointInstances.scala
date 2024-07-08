package at.ac.oeaw.imba.gerlich.looptrace
package instances

import cats.*
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt

trait ImagingTimepointInstances:
    import NonnegativeInt.given
    given showForImagingTimepoint(using ev: Show[NonnegativeInt]): Show[ImagingTimepoint] = 
        ev.contramap(_.get)
    given SimpleShow[ImagingTimepoint] = SimpleShow.fromShow
