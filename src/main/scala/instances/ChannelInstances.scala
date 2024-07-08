package at.ac.oeaw.imba.gerlich.looptrace
package instances

import cats.syntax.all.*

import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt

trait ChannelInstances:
    import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt.given
    given simpleShowForChannel(using ev: SimpleShow[NonnegativeInt]): SimpleShow[Channel] = ev.contramap(_.get)
