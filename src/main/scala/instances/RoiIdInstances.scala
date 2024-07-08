package at.ac.oeaw.imba.gerlich.looptrace
package instances

import cats.*
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt

trait RoiIdInstances:
    given simpleShowForRoiIndex(using ev: SimpleShow[NonnegativeInt]): SimpleShow[RoiIndex] = 
        ev.contramap(_.get)
