package at.ac.oeaw.imba.gerlich.looptrace
package instances

import cats.*
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt

/** Typeclass instances for [[at.ac.oeaw.imba.gerlich.looptrace.RoiIndex]] */
trait RoiIdInstances:
    /** Show a [[at.ac.oeaw.imba.gerlich.looptrace]] by string representation of the wrapped integer. */
    given (ev: SimpleShow[NonnegativeInt]) => SimpleShow[RoiIndex] = ev.contramap(_.get)
