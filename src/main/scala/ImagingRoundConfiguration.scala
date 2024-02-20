package at.ac.oeaw.imba.gerlich.looptrace

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
    import RegionalGrouping.Semantic.*
    
    /** Error for when building an instance fails */
    class BuildError(val messages: NonEmptyList[String]) 
        extends Exception(s"Error(s) creating imaging round configuration: ${messages.mkString_(", ")}")

    /** Check that one set of timepoints is a subset of another */
    def checkTimesSubset(knownTimes: Set[Timepoint])(times: Set[Timepoint], context: String): ValidatedNel[String, Unit] = (times -- knownTimes).toList match {
        case Nil => ().validNel
        case unknown => s"Unknown timepoint(s) in $context: ${unknown.map(_.show).mkString(", ")}".invalidNel
    }

    /**
      * Validate construction of the imaging round configuration by checking that all timepoints are known and covered.
      * 
      * More specifically, any timepoint to exclude from the tracing must exist as a timepoint in the sequence of imaging 
      * rounds, regardless of whether the grouping of regional timepoints is trivial or not.
      * 
      * Furthermore, if the regional grouping is nontrivial, then the set of timepoints in the regional grouping must be 
      * both a subset and a superset of the set of timepoints from regional rounds in the imaging sequence. 
      * In other words, those sets of timepoints must be equivalent.
      *
      * @param sequence The declaration of sequential FISH rounds that define an experiment
      * @param grouping How to group regional FISH rounds for proximity filtration
      * @param tracingExclusions Timepoints to exclude from tracing analysis
      * @return
      */
    def build(sequence: ImagingSequence, grouping: RegionalGrouping, tracingExclusions: Set[Timepoint]): ErrMsgsOr[ImagingRoundConfiguration] = {
        val knownTimes = sequence.rounds.map(_.time).toList.toSet
        // Regardless of the subtype of grouping, we need to check that any tracing exclusion timepoint is a known timepoint.
        val tracingSubsetNel = checkTimesSubset(knownTimes)(tracingExclusions, "tracing exclusions")
        (grouping match {
            // TODO: need to use just the regional timepoints from the sequence.
            case g: RegionalGrouping.Trivial.type => 
                // In the trivial grouping case, we have no more validation work to do.
                tracingSubsetNel.map(_ => ImagingRoundConfiguration(sequence, g, tracingExclusions))
            case g: RegionalGrouping.Nontrivial => 
                // When the grouping's nontrivial, check for set equivalance of timepoints b/w imaging sequence and regional grouping.
                val groupedTimes = g.groups.reduce(_ ++ _).toList.toSet
                val groupingSubsetNel = checkTimesSubset(knownTimes)(groupedTimes, "regional grouping (rel. to imaging sequence)")
                val groupingSupersetNel = checkTimesSubset(groupedTimes)(knownTimes, "imaging sequence (rel. to regional grouping)")
                (tracingSubsetNel, groupingSubsetNel, groupingSupersetNel).tupled.map(_ => 
                    ImagingRoundConfiguration(sequence, grouping, tracingExclusions)
                )
        }).toEither
    }

    /**
      * Read the configuration of imaging rounds for the experiment, including regional grouping and 
      * exclusions from tracing.
      *
      * @param jsonFile Path to the file from which to parse the configuration
      * @return Either a [[scala.util.Left]]-wrapped nonempty collection of error messages, or 
      *     a [[scala.util.Right]]-wrapped, successfully parsed configuration instance
      */
    def fromJsonFile(jsonFile: os.Path): ErrMsgsOr[ImagingRoundConfiguration] = 
        Try{ readJsonFile[ujson.Value](jsonFile) }
            .toEither
            .leftMap(e => NonEmptyList.one(e.getMessage))
            .flatMap(fromJson)
    
    /** Try to read a configuration directly from JSON. */
    def fromJson(fullJsonData: ujson.Value): ErrMsgsOr[ImagingRoundConfiguration] = 
        safeReadAs[Map[String, ujson.Value]](fullJsonData).leftMap(NonEmptyList.one).flatMap(fromJsonMap)

    /** Attempt to parse a configuration from a key-value mapping from section name to JSON value. */
    def fromJsonMap(data: Map[String, ujson.Value]): ErrMsgsOr[ImagingRoundConfiguration] = {
        given rwForRound: Reader[ImagingRound] = ImagingRound.rwForImagingRound
        val roundsNel: ValidatedNel[String, ImagingSequence] = 
            data.get("imagingRounds")
                .toRight("Missing imagingRounds key!")
                .flatMap(safeReadAs[List[ImagingRound]])
                .leftMap(NonEmptyList.one)
                .flatMap(ImagingSequence.fromRounds)
                .toValidated
        val crudeGroupingNel: ValidatedNel[String, Option[(RegionalGrouping.Semantic, NonEmptyList[NonEmptySet[Timepoint]])]] = 
            data.get("regionalGrouping") match {
                case None => Validated.Valid(None)
                case Some(fullJson) => safeReadAs[Map[String, ujson.Value]](fullJson).leftMap(NonEmptyList.one).flatMap{ currentSection => 
                    val semanticNel = currentSection.get("semantic")
                        .toRight("Missing semantic in regional grouping section!")
                        .flatMap(safeReadAs[String](_).flatMap{
                            case ("permissive" | "Permissive" | "PERMISSIVE") => Permissive.asRight
                            case ("prohibitive" | "Prohibitive" | "PROHIBITIVE") => Prohibitive.asRight
                            case s => s"Illegal value for regional grouping semantic: $s".asLeft
                        })
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
        (roundsNel, crudeGroupingNel, tracingExclusionsNel).tupled.toEither.flatMap{ case (sequence, maybeCrudeGrouping, exclusions) =>
            val regionalTimepoints = sequence.rounds.toList.flatMap{ 
                case (r: RegionalImagingRound) => r.time.some
                case _ => None
            }.toSet
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
                    val existenceNel = { // Every regional timepoint in the grouping must exist in the experiment.
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
                    (existenceNel, disjointnessNel)
                        .tupled
                        .toEither
                        .map(Function.const{ semantic match {
                            case Permissive => RegionalGrouping.Permissive(grouping)
                            case Prohibitive => RegionalGrouping.Prohibitive(grouping)
                        }}).toValidated
            }
            val uncoveredRounds: ValidatedNel[String, Unit] = maybeCrudeGrouping match {
                // If regional grouping is present, some regional rounds in the experiment may not have been covered.
                case None => ().validNel[String]
                case Some((_, grouping)) => 
                    val groupedTimepoints: Set[Timepoint] = grouping.reduce(_ ++ _).toSortedSet
                    regionalTimepoints.filterNot(groupedTimepoints.contains).toList.toNel.toLeft(()).leftMap{ uncovered => 
                        val timeText = uncovered.map(_.show).mkString_(", ")
                        s"Timepoint(s) in experiment not covered by regional grouping: $timeText"
                    }.toValidatedNel
            }
            (nonexistentExclusionsNel, groupingNel, uncoveredRounds)
                .mapN((_, grouping, _) => ImagingRoundConfiguration(sequence, grouping, exclusions))
                .toEither
        }
    }

    /**
     * Create instance, throw exception if any failure occurs
     * 
     * @see [[ImagingRoundConfiguration.build]]
     */
    def unsafe(sequence: ImagingSequence, grouping: RegionalGrouping, tracingExclusions: Set[Timepoint]): ImagingRoundConfiguration = 
        build(sequence, grouping, tracingExclusions).fold(messages => throw new BuildError(messages), identity)

    /**
     * Create instance, throw exception if any failure occurs
     * 
     * @see [[ImagingRoundConfiguration.build]]
     */
    def unsafe(
        sequence: ImagingSequence, 
        maybeGrouping: Option[(RegionalGrouping.Semantic, RegionalGrouping.Groups)], 
        tracingExclusions: Set[Timepoint],
    ): ImagingRoundConfiguration = {
        val grouping = maybeGrouping.fold(RegionalGrouping.Trivial)(RegionalGrouping.Nontrivial.apply.tupled)
        unsafe(sequence, grouping, tracingExclusions)
    }

    /** Alias to give more context-rich meaning to a nonempty collection of timepoints */
    type RegionalImageRoundGroup = NonEmptySet[Timepoint]

    /** Helpers for working with timepoint groupings */
    object RegionalImageRoundGroup:
        given rwForRegionalImageRoundGroup: ReadWriter[RegionalImageRoundGroup] = readwriter[ujson.Value].bimap(
            group => ujson.Arr(group.toList.map(name => ujson.Num(name.get))*), 
            json => json.arr
                .toList
                .toNel
                .toRight("Empty collection can't parse as group of regional imaging rounds!")
                .flatMap(_.traverse(_.safeInt.flatMap(Timepoint.fromInt)))
                .flatMap(safeNelToNes)
                .fold(
                    repeats => throw new ujson.Value.InvalidData(json, s"Repeat values for group of regional imaging rounds: $repeats"), 
                    identity
                    )
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
        /** Helpers for constructing and owrking with a nontrivial grouping of regional imaging rounds */
        object Nontrivial:
            /** Dispatch to the appropriate leaf class constructor based on the value of the semantic. */
            def apply(semantic: Semantic, groups: Groups): Nontrivial = semantic match {
                case Semantic.Permissive => Permissive(groups)
                case Semantic.Prohibitive => Prohibitive(groups)
            }
            /** Construct a grouping with a single group. */
            def singleton(semantic: Semantic, group: RegionalImageRoundGroup): Nontrivial = apply(semantic, NonEmptyList.one(group))
        end Nontrivial
        /** A 'permissive' grouping 'allows' members of the same group to violate some rule, while 'forbidding' non-grouped items from doing so. */
        final case class Permissive(groups: Groups) extends Nontrivial
        /** Helpers for working with the permissive regional grouping */
        object Permissive:
            /** Construct a grouping with a single group. */
            def singleton(group: RegionalImageRoundGroup): Permissive = Permissive(NonEmptyList.one(group))
        end Permissive
        /** A 'prohibitive' grouping 'forbids' members of the same group to violate some rule, while 'allowing' non-grouped items to violate the rule. */
        final case class Prohibitive(groups: Groups) extends Nontrivial
        /** Helpers for working with the prohibitive regional grouping */
        object Prohibitive:
            /** Construct a grouping with a single group. */
            def singleton(group: RegionalImageRoundGroup): Prohibitive = Prohibitive(NonEmptyList.one(group))
        end Prohibitive

        /** Delineate which semantic is desired */
        private[looptrace] enum Semantic:
            case Permissive, Prohibitive
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