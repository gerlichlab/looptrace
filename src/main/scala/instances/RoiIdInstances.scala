package at.ac.oeaw.imba.gerlich.looptrace
package instances

import cats.*
import cats.syntax.all.*
import io.github.iltotore.iron.:|
import io.github.iltotore.iron.constraint.any.Not
import io.github.iltotore.iron.constraint.numeric.Negative
import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow

/** Typeclass instances for [[at.ac.oeaw.imba.gerlich.looptrace.RoiIndex]] */
trait RoiIdInstances:
  /** Show a [[at.ac.oeaw.imba.gerlich.looptrace]] by string representation of
    * the wrapped integer.
    */
  given (ev: SimpleShow[Int :| Not[Negative]]) => SimpleShow[RoiIndex] =
    ev.contramap(_.get)
