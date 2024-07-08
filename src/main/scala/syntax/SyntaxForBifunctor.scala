package at.ac.oeaw.imba.gerlich.looptrace
package syntax

import cats.Bifunctor
import cats.syntax.bifunctor.*

trait SyntaxForBifunctor:
    extension [A, F[_, _] : Bifunctor](faa: F[A, A])
        def mapBoth[B](f: A => B): F[B, B] = faa.bimap(f, f)

