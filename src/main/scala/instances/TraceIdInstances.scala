package at.ac.oeaw.imba.gerlich.looptrace
package instances

import cats.*
import cats.syntax.all.*
import io.github.iltotore.iron.:|
import io.github.iltotore.iron.constraint.any.Not
import io.github.iltotore.iron.constraint.numeric.Negative
import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow

/** Basic typeclass instances for [[TraceId]] */
trait TraceIdInstances:
  given (ev: SimpleShow[Int :| Not[Negative]]) => SimpleShow[TraceId] =
    ev.contramap(_.get)

  given (ev: SimpleShow[String]) => SimpleShow[TraceGroupMaybe] =
    ev.contramap(_.toOption.fold("")(_.get))
end TraceIdInstances
