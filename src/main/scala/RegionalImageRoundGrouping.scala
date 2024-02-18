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
            .toRight("Empty collection can't parse as probe group!")
            .flatMap(_.traverse(_.safeInt.flatMap(Timepoint.fromInt)))
            .flatMap(safeNelToNes)
            .leftMap(repeats => s"Repeat values for probe group: $repeats")
            .fold(msg => throw new ujson.Value.InvalidData(json, msg), RegionalImageRoundGroup.apply)
    )
end RegionalImageRoundGroup

/** How to permit or prohibit regional barcode imaging probes/timepoints from being too physically close */
sealed trait RegionalImageRoundGrouping

object RegionalImageRoundGrouping:
    case object Trivial extends RegionalImageRoundGrouping
    sealed trait Nontrivial extends RegionalImageRoundGrouping:
        def groups: List[RegionalImageRoundGroup]
    final case class Permissive(groups: List[RegionalImageRoundGroup]) extends Nontrivial
    final case class Prohibitive(groups: List[RegionalImageRoundGroup]) extends Nontrivial
end RegionalImageRoundGrouping
