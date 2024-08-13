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

    /**
      * Partition a collection of objects into {@code n} distinct parts (subsets).
      * 
      * @tparam X The type of object in the collection to partition
      * @param n The number of (disjoint) subsets with which to cover the input set
      * @param xs The input set to partition
      * @return A collection in which each element is a disjoint covering (i.e., a partition) 
      *         of the given input collection
      */
    private[looptrace] def partition[X](n: Int, xs: Set[X]): List[List[Set[X]]] = {
        require(n > 0, s"Desired number of subsets must be strictly postitive, not $n")
        require(n <= xs.size, s"Desired number of subsets exceeds number of elements: $n > ${xs.size}")
        powerset(xs)
            .filter(_.nonEmpty)
            .combinations(n)
            .foldRight(List.empty[List[Set[X]]]){ (subs, acc) =>
                val part = subs.map(_.toSet)
                if part.map(_.size).sum === xs.size && part.combineAll === xs then (part :: acc) else acc
            }
    }

    private def powerset[X](xs: Set[X]): List[Seq[X]] = List.range(0, xs.size + 1).flatMap(xs.toSeq.combinations(_).toList)
end collections
