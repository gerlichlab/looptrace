package at.ac.oeaw.imba.gerlich.looptrace
package syntax

import scala.util.Try
import cats.data.ValidatedNel
import cats.syntax.all.*

trait SyntaxForTry:
    extension [A](t: Try[A])
        def toValidatedNel: ValidatedNel[Throwable, A] = t.toEither.toValidatedNel
