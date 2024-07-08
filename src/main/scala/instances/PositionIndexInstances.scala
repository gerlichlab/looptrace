package at.ac.oeaw.imba.gerlich.looptrace
package instances

import cats.*
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

trait PositionIndexInstances:
    given simpleShowForPositionIndex(using ev: SimpleShow[NonnegativeInt]): SimpleShow[PositionIndex] = 
        ev.contramap(_.get)
