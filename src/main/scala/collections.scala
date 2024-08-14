package at.ac.oeaw.imba.gerlich.looptrace

import scala.collection.immutable.SortedSet
import cats.*
import cats.data.*
import cats.syntax.all.*

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
end collections
