package at.ac.oeaw.imba.gerlich.looptrace

import cats.syntax.all.*

/** Tools for working with collections */
package object collections:
    /**
      * Take the cartesian product over the given input collections.
      *
      * @tparam X The element type
      * @param xss The collection of collections over which to take the cartesian product
      * @return The cartesian product over the given input
      */
    def cartesianProduct[X](xss: Seq[Seq[X]]): Seq[Seq[X]] = {
        @scala.annotation.tailrec
        def go(rest: Seq[Seq[X]], acc: Seq[Seq[X]]): Seq[Seq[X]] = rest match {
            case Nil => acc
            // Prepend each current element to each (leftward-growing) output collection, starting a new "branch" for each new element.
            case curr :: rest => go(rest, curr.flatMap{ x => acc.map(x +: _) })
        }
        // Flip the input since deconstruction will be left-to-right, but we'll be PRE-pending to accumulator.
        // Initialise the accumulator with the terminus of an empty collection (itself a collection).
        go(xss.reverse, Seq(Nil))
    }

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
