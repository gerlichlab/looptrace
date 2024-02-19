package at.ac.oeaw.imba.gerlich.looptrace

import scala.language.adhocExtensions // for extending ujson.Value.InvalidData
import scala.util.Try
import cats.*
import cats.data.{ NonEmptyList, NonEmptySet, Validated, ValidatedNel }
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.{ readJsonFile, safeReadAs }

/** Something which admits--at minimum--a sequence of imaging rounds */
trait ImagingRoundConfigurationLike:
    def getImagingRoundSequence: ImagingSequence
end ImagingRoundConfigurationLike

/** Typical looptrace declaration/configuration of imaging rounds and how to use them */
final case class ImagingRoundConfiguration private(
    sequenceOfRounds: ImagingSequence, 
    regionalGrouping: ImagingRoundConfiguration.RegionalGrouping, 
    // TODO: by default, skip regional and blank imaging rounds (but do use repeats).
    notForTracing: Set[Timepoint], // Timepoints of imaging rounds to not use for tracing
    )

/** Tools for working with declaration of imaging rounds and how to use them within an experiment */
object ImagingRoundConfiguration:
    final case class DecodingError(messages: NonEmptyList[String], json: ujson.Value)
    
    /**
      * Read the configuration of imaging rounds for the experiment, including regional grouping and 
      * exclusions from tracing.
      *
      * @param jsonFile Path to the file from which to parse the configuration
      * @return Either a [[scala.util.Left]]-wrapped nonempty collection of error messages, or 
      *     a [[scala.util.Right]]-wrapped, successfully parsed configuration instance
      */
    def fromJsonFile(jsonFile: os.Path): ErrMsgsOr[ImagingRoundConfiguration] = {
        given rwForRound: Reader[ImagingRound] = ImagingRound.rwForImagingRound
        Try{ readJsonFile[Map[String, ujson.Value]](jsonFile) }
            .toEither
            .leftMap(e => NonEmptyList.one(e.getMessage)) // some error even getting key-value pairs
            .flatMap{ data =>
                val roundsNel: ValidatedNel[String, ImagingSequence] = 
                    data.get("imagingRounds")
                        .toRight("Missing imagingRounds key!")
                        .flatMap(safeReadAs[List[ImagingRound]](_))
                        .leftMap(NonEmptyList.one)
                        .flatMap(ImagingSequence.fromRounds)
                        .toValidated
                val crudeGroupingNel: ValidatedNel[String, Option[(String, NonEmptyList[NonEmptySet[Timepoint]])]] = 
                    data.get("regionalGrouping") match {
                        case None => Validated.Valid(None)
                        case Some(fullJson) => safeReadAs[Map[String, ujson.Value]](fullJson).leftMap(NonEmptyList.one).flatMap{ currentSection => 
                            val semanticNel = currentSection.get("semantic")
                                .toRight("Missing semantic in regional grouping section!")
                                .flatMap(safeReadAs[String])
                                .toValidatedNel
                            val groupsNel = currentSection.get("groups")
                                .toRight("Missing groups in regional grouping section!")
                                .flatMap(safeReadAs[List[List[Int]]])
                                .flatMap(_.traverse(_.traverse(Timepoint.fromInt))) // Lift all ints to timepoints.
                                .flatMap(liftToNel(_, "regional grouping".some)) // Entire collection must be nonempty.
                                .flatMap(_.traverse(liftToNes(_, "regional group".some))) // Each group must be nonempty.
                                .toValidatedNel
                            (semanticNel, groupsNel).tupled.toEither.map(_.some)
                        }.toValidated
                    }
                val tracingExclusionsNel: ValidatedNel[String, Set[Timepoint]] = 
                    data.get("regionalGrouping") match {
                        case None => Validated.Valid(Set())
                        case Some(json) => safeReadAs[List[Int]](json)
                            .flatMap(_.traverse(Timepoint.fromInt))
                            .map(_.toSet)
                            .toValidatedNel
                    }
                (roundsNel, crudeGroupingNel, tracingExclusionsNel).tupled.toEither
            }
            .flatMap{ case (sequence, maybeCrudeGrouping, exclusions) =>
                // Some of the timepoints specified for tracing exclusion may not correspond to extant imaging rounds.
                val nonexistentExclusionsNel: ValidatedNel[String, Unit] = 
                    (exclusions -- sequence.rounds.map(_.time).toList)
                        .toList
                        .toNel
                        .toLeft(())
                        .leftMap(ts => s"Timepoint(s) to exclude from tracing aren't in imaging sequence: ${ts.map(_.show).mkString_(", ")}")
                        .toValidatedNel
                // Some of the timepoints specified in regional grouping may not exist as regional imaging rounds.
                val groupingNel: ValidatedNel[String, RegionalGrouping] = maybeCrudeGrouping match {
                    case None => Validated.Valid(RegionalGrouping.Trivial)
                    case Some((semantic, grouping)) => 
                        val existenceNel = {
                            val regionalTimepoints = sequence.rounds.toList.flatMap{ 
                                case (r: RegionalImagingRound) => r.time.some
                                case _ => None
                                }.toSet
                            grouping.reduce(_ ++ _).filterNot(regionalTimepoints.contains).toList.toNel match {
                                case None => ().validNel[String]
                                case Some(nonRegExcl) => 
                                    val timeText = nonRegExcl.map(_.show).mkString_(", ")
                                    s"Timepoint(s) in regional grouping don't exist or aren't regional: $timeText".invalidNel[Unit]
                            }
                        }
                        val disjointnessNel = grouping.toList.combinations(2).toList.flatMap{
                            case g1 :: g2 :: Nil => (g1 & g2).toNes
                            case gs => throw new IllegalStateException(s"Got ${gs.size} groups (not 2!) when taking pairs!")
                        }.foldLeft(Set.empty[Timepoint])(_ ++ _.toSortedSet).toList.toNel.toLeft(())
                            .leftMap{ overlaps => s"Overlap(s) among regional grouping: ${overlaps.map(_.show).mkString_(", ")}" }
                            .toValidatedNel
                        (existenceNel, disjointnessNel).tupled.toEither.flatMap( Function.const{
                            val unwrappedGrouping = grouping.map(RegionalImageRoundGroup.apply)
                            semantic match {
                                case "permissive" => RegionalGrouping.Permissive(unwrappedGrouping).asRight
                                case "prohibitive" => RegionalGrouping.Prohibitive(unwrappedGrouping).asRight
                                case _ => NonEmptyList.one(s"Illegal grouping semantic: $semantic").asLeft
                            }
                        } ).toValidated
                }
                // If regional grouping is present, some regional rounds in the experiment may not have been covered.
                val uncoveredRounds: ValidatedNel[String, Unit] = ???
                (nonexistentExclusionsNel, groupingNel, uncoveredRounds)
                    .mapN((_, grouping, _) => ImagingRoundConfiguration(sequence, grouping, exclusions))
                    .toEither
            }
    }

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
    sealed trait RegionalGrouping

    /** The (concrete) subtypes of regional image round grouping */
    object RegionalGrouping:
        type Groups = NonEmptyList[RegionalImageRoundGroup]
        /** A trivial grouping of regional imaging rounds, which treats all regional rounds as one big group */
        case object Trivial extends RegionalGrouping
        /** A nontrivial grouping of regional imaging rounds, which must constitute a partition of those available  */
        sealed trait Nontrivial extends RegionalGrouping:
            /** A nontrivial grouping specifies a list of groups which comprise the total grouping.s */
            def groups: Groups
        /** A 'permissive' grouping 'allows' members of the same group to violate some rule, while 'forbidding' non-grouped items from doing so. */
        final case class Permissive private[ImagingRoundConfiguration](groups: Groups) extends Nontrivial
        /** A 'prohibitive' grouping 'forbids' members of the same group to violate some rule, while 'allowing' non-grouped items to violate the rule. */
        final case class Prohibitive private[ImagingRoundConfiguration](groups: Groups) extends Nontrivial
    end RegionalGrouping

    /** Check list of items for nonemptiness. */
    private def liftToNel[A](as: List[A], context: Option[String] = None): Either[String, NonEmptyList[A]] = 
        as.toNel.toRight(context.fold("Empty list!")(ctx => s"List for $ctx is empty!"))
    
    /** Lift a list of items into a nonempty set, checking for uniqueness and nonemptiness. */
    private def liftToNes[A : Order](as: List[A], context: Option[String] = None): Either[String, NonEmptySet[A]] = 
        liftToNel(as, context).flatMap(nel => 
            val candidate = nel.toNes
            val unique = candidate.size.toInt
            val total = nel.length
            (unique === total).either(
                s"$total total - $unique unique = ${total - unique} repeated items" ++ context.fold("")(ctx => s"for $ctx"), 
                candidate
                )
        )
end ImagingRoundConfiguration