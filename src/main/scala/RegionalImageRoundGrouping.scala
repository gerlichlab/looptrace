package at.ac.oeaw.imba.gerlich.looptrace

import cats.data.NonEmptySet
import cats.syntax.all.*
import upickle.default.*

/**
 * Designation of regional barcode timepoints which are prohibited from being in (configurably) close proximity.
 *
 * @param get The actual collection of indices
 */
final case class RegionalImageRoundGroup(get: NonEmptySet[Timepoint])

/** Helpers for working with timepoint groupings */
object RegionalImageRoundGroup:
    given rwForRegionalImageRoundGroup: ReadWriter[RegionalImageRoundGroup] = readwriter[ujson.Value].bimap(
        group => ujson.Arr(group.get.toList.map(name => ujson.Num(name.get))*), 
        json => json.arr
            .toList
            .toNel
            .toRight("Empty collection can't parse as group of regional imaging rounds!")
            .flatMap(_.traverse(_.safeInt.flatMap(Timepoint.fromInt)))
            .flatMap(safeNelToNes)
            .leftMap(repeats => s"Repeat values for group of regional imaging rounds: $repeats")
            .fold(msg => throw new ujson.Value.InvalidData(json, msg), RegionalImageRoundGroup.apply)
    )
end RegionalImageRoundGroup

/** How to permit or prohibit regional barcode imaging probes/timepoints from being too physically close */
sealed trait RegionalImageRoundGrouping

/** The (concrete) subtypes of regional image round grouping */
object RegionalImageRoundGrouping:
    /** A trivial grouping of regional imaging rounds, which treats all regional rounds as one big group */
    case object Trivial extends RegionalImageRoundGrouping
    /** A nontrivial grouping of regional imaging rounds, which must constitute a partition of those available  */
    sealed trait Nontrivial extends RegionalImageRoundGrouping:
        /** A nontrivial grouping specifies a list of groups which comprise the total grouping.s */
        def groups: List[RegionalImageRoundGroup]
    /** A 'permissive' grouping 'allows' members of the same group to violate some rule, while 'forbidding' non-grouped items from doing so. */
    final case class Permissive(groups: List[RegionalImageRoundGroup]) extends Nontrivial
    /** A 'prohibitive' grouping 'forbids' members of the same group to violate some rule, while 'allowing' non-grouped items to violate the rule. */
    final case class Prohibitive(groups: List[RegionalImageRoundGroup]) extends Nontrivial
end RegionalImageRoundGrouping
