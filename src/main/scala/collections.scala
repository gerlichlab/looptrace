package at.ac.oeaw.imba.gerlich.looptrace

import scala.collection.immutable.SortedSet
import scala.collection.SeqOps
import cats.*
import cats.data.*
import cats.syntax.all.*
import io.github.iltotore.iron.{ :|, Constraint, refineEither, refineUnsafe }
import io.github.iltotore.iron.constraint.collection.MinLength

/** Tools for working with collections */
package object collections:
    /** Syntax helpers for working with various Set implementations */
    extension [X : Ordering](xs: Set[X])
        def toNonEmptySet: Option[NonEmptySet[X]] = NonEmptySet.fromSet(xs.toSortedSet)
        def toNonEmptySetUnsafe: NonEmptySet[X] = xs.toNonEmptySet.getOrElse {
            throw new IllegalArgumentException("Cannot create NonEmptySet from empty set")
        }
        def toSortedSet: SortedSet[X] = SortedSet.from(xs)

    extension [X](xs: NonEmptySet[X])
        def ++(ys: Set[X])(using Ordering[X]): NonEmptySet[X] = ys.toNonEmptySet.fold(xs)(xs | _)

    type AtLeast2[C[*], E] = C[E] :| MinLength[2]

    object AtLeast2:
        def apply[X](x: X, xs: NonEmptyList[X]): AtLeast2[List, X] = 
            (x :: xs).toList.refineUnsafe[MinLength[2]]
        
        inline def either[C[*], X](xs: C[X])(using inline: Constraint[C[X], MinLength[2]]): Either[String, AtLeast2[C, X]] = 
            xs.refineEither[MinLength[2]]
        
        extension [X](xs: AtLeast2[Set, X])
            infix def +(x: X): AtLeast2[Set, X] = 
                (xs + x).refineUnsafe[MinLength[2]]

        private type AtLeast2FixedC[C[*]] = [X] =>> AtLeast2[C, X]
        
        given functorForAtLeast2[C[*]: Functor]: Functor[AtLeast2FixedC[C]] with
            override def map[A, B](fa: AtLeast2FixedC[C][A])(f: A => B): AtLeast2FixedC[C][B] = 
                fa.map(f)
    end AtLeast2
end collections
