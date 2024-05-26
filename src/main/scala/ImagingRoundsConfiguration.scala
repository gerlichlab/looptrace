package at.ac.oeaw.imba.gerlich.looptrace

import scala.collection.immutable.SortedSet
import scala.language.adhocExtensions // to extend ujson.Value.InvalidData
import scala.util.Try
import cats.*
import cats.data.{ NonEmptyList, NonEmptyMap, NonEmptySet, Validated, ValidatedNel }
import cats.data.Validated.{ Invalid, Valid }
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*
import com.typesafe.scalalogging.LazyLogging
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.{ readJsonFile, safeReadAs }
import at.ac.oeaw.imba.gerlich.looptrace.space.{ DistanceThreshold, PiecewiseDistance }
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.LocusGroup

/** Typical looptrace declaration/configuration of imaging rounds and how to use them */
final case class ImagingRoundsConfiguration private(
    sequence: ImagingSequence, 
    locusGrouping: Set[ImagingRoundsConfiguration.LocusGroup], // should be empty iff there are no locus rounds in the sequence
    proximityFilterStrategy: ImagingRoundsConfiguration.ProximityFilterStrategy,
    // TODO: We could, by default, skip regional and blank imaging rounds (but do use repeats).
    tracingExclusions: Set[Timepoint], // Timepoints of imaging rounds to not use for tracing
    ):
    
    /** Simply take the rounds from the contained imagingRounds sequence. */
    final def allRounds: NonEmptyList[ImagingRound] = sequence.allRounds
    
    /** The number of imaging rounds is the length of the imagingRounds sequence. */
    final def numberOfRounds: Int = sequence.length
    
    /** Only compute this when necessary, but retain as val since memory footprint 
     * will be small (not so many imaging rounds), for faster ordered iteration */
    private lazy val regionTimeToLocusTimes: NonEmptyMap[Timepoint, SortedSet[Timepoint]] = (
        locusGrouping.toList.toNel match {
            case None => sequence.regionRounds.map{ rr => rr.time -> SortedSet(sequence.locusRounds.map(_.time)*) }
            case Some(groups) => groups.map{ case LocusGroup(rt, lts) => rt -> lts.toSortedSet }
        }
    ).toNem
    
    /** Faciliate lookup of reindexed timepoint for visualisation. */
    lazy val lookupReindexedTimepoint: Map[Timepoint, Map[Timepoint, Int]] = {
        // First, group timepoints by regional timepoint.
        val sets = locusGrouping.toList.toNel match {
            case None => 
                // When the locus grouping is absent/empty, then for each regional timepoint 
                // we're interested in the full sequence of imaging timepoints from the experiment.
                sequence.regionRounds.map{ rr => rr.time -> SortedSet(sequence.locusRounds.map(_.time)*) }
            case Some(groups) => 
                // If the locusGrouping is nonempty, then the timepoints associated with each 
                // regional timepoint are its locus timepoints and the regional timepoint itself.
                groups.map{ case LocusGroup(rt, lts) => rt -> (lts.toSortedSet + rt) }
        }
        sets.toList.map{ (rt, lts) => rt -> lts.toList.zipWithIndex.toMap }.toMap
    }
end ImagingRoundsConfiguration

/** Tools for working with declaration of imaging rounds and how to use them within an experiment */
object ImagingRoundsConfiguration extends LazyLogging:
    
    /** Something went wrong with attempt to instantiate a configuration */
    trait BuildErrorLike:
        def messages: NonEmptyList[String]
    end BuildErrorLike

    /** Construct errors during instantiation */
    object BuildError:
        /** An error building a configuration from pure values */
        final class FromPure(val messages: NonEmptyList[String]) 
            extends Exception(condenseMessages(messages, None)) with BuildErrorLike
        final class FromJsonFile(val messages: NonEmptyList[String], file: os.Path)
            extends Exception(condenseMessages(messages, s"(from file $file)".some)) with BuildErrorLike
        /** An error building a configuration from JSON */
        final class FromJson(val messages: NonEmptyList[String], json: ujson.Value)  
            extends ujson.Value.InvalidData(json, condenseMessages(messages, None)) with BuildErrorLike
        /** Combine potentially multiple error messages into one */
        private def condenseMessages(messages: NonEmptyList[String], extraContext: Option[String]): String = 
            s"Error(s) building imaging round configuration:" ++ extraContext.fold("")(" " ++ _ ++ " ") ++ messages.mkString_(", ")
    end BuildError

    /** Check that one set of timepoints is a subset of another */
    def checkTimesSubset(knownTimes: Set[Timepoint])(times: Set[Timepoint], context: String): ValidatedNel[String, Unit] = 
        (times -- knownTimes).toList match {
            case Nil => ().validNel
            case unknown => s"Unknown timepoint(s) ($context): ${unknown.sorted.map(_.show).mkString(", ")}".invalidNel
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
      * @param locusGrouping How each locus is associated with a region
      * @param proximityFilterStrategy How to filter regional spots based on proximity
      * @param tracingExclusions Timepoints to exclude from tracing analysis
      * @param checkLocusTimepointCovering Whether to validate that the union of the locusGrouping section values covers all locus imaging timepoints
      * @return Either a [[scala.util.Left]]-wrapped nonempty list of error messages, or a [[scala.util.Right]]-wrapped built instance
      */
    def build(sequence: ImagingSequence, locusGrouping: Set[LocusGroup], proximityFilterStrategy: ProximityFilterStrategy, tracingExclusions: Set[Timepoint], checkLocusTimepointCovering: Boolean): ErrMsgsOr[ImagingRoundsConfiguration] = {
        val knownTimes = sequence.allTimepoints
        // Regardless of the subtype of proximityFilterStrategy, we need to check that any tracing exclusion timepoint is a known timepoint.
        val tracingSubsetNel = checkTimesSubset(knownTimes.toSortedSet)(tracingExclusions, "tracing exclusions")
        // TODO: consider checking that every regional timepoint in the sequence is represented in the locusGrouping.
        // See: https://github.com/gerlichlab/looptrace/issues/270
        val uniqueTimepointsInLocusGrouping = locusGrouping.map(_.locusTimepoints).foldLeft(Set.empty[Timepoint])(_ ++ _.toSortedSet)
        val (locusTimeSubsetNel, locusTimeSupersetNel) = {
            if locusGrouping.isEmpty then (().validNel, ().validNel) else {
                val locusTimesInSequence = sequence.locusRounds.map(_.time).toSet
                // First, check that the union of values in the locus grouping is a subset of the locus-specific imaging rounds.
                val subsetNel = (uniqueTimepointsInLocusGrouping -- locusTimesInSequence).toList match {
                    case Nil => ().validNel
                    case ts => s"${ts.length} timepoint(s) in locus grouping and not found as locus imaging timepoints: ${ts.sorted.map(_.get).mkString(", ")}".invalidNel
                }
                // Then, check that each locus-specific imaging round is in the locus grouping, as appropriate..
                val supersetNel = 
                    if checkLocusTimepointCovering
                    then
                        val missing = locusTimesInSequence -- uniqueTimepointsInLocusGrouping
                        logger.debug(s"${missing.size} locus imaging timepoint(s) missing from the locus grouping: ${mkStringTimepoints(missing)}")
                        val (correctlyMissing, wronglyMissing) = missing.partition(tracingExclusions.contains)
                        logger.debug(s"${correctlyMissing.size} locus imaging timepoint(s) CORRECTLY missing from locus grouping: ${mkStringTimepoints(correctlyMissing)}")
                        if wronglyMissing.isEmpty 
                        then ().validNel
                        else s"${wronglyMissing.size} locus timepoint(s) in imagingRounds and not found in locusGrouping (nor in tracingExclusions): ${mkStringTimepoints(wronglyMissing)}".invalidNel
                    else
                        ().validNel
                (subsetNel, supersetNel)
            }
        }
        val locusGroupTimesAreRegionTimesNel = {
            val nonRegional = locusGrouping.map(_.regionalTimepoint) -- sequence.regionRounds.map(_.time).toList
            if nonRegional.isEmpty then ().validNel 
            else s"${nonRegional.size} timepoint(s) as keys in locus grouping that aren't regional.".invalidNel
        }
        val (proximityGroupingSubsetNel, proximityGroupingSupersetNel) = proximityFilterStrategy match {
            case (UniversalProximityPermission | UniversalProximityProhibition(_)) => 
                // In the trivial case, we have no more validation work to do.
                (().validNel, ().validNel)
            case s: (SelectiveProximityPermission | SelectiveProximityProhibition) => 
                // In the nontrivial case, check for set equivalance of timepoints b/w imaging sequence and grouping.
                val uniqueGroupedTimes = s.grouping.reduce(_ ++ _).toList.toSet
                val uniqueRegionalTimes = sequence.regionRounds.map(_.time).toList.toSet
                val subsetNel = checkTimesSubset(uniqueRegionalTimes)(uniqueGroupedTimes, "proximity filter's grouping (rel. to regionals in imaging sequence)")
                val supersetNel = checkTimesSubset(uniqueGroupedTimes)(uniqueRegionalTimes, "regionals in imaging sequence (rel. to proximity filter strategy)")
                (subsetNel, supersetNel)
        }
        (tracingSubsetNel, locusTimeSubsetNel, locusTimeSupersetNel, locusGroupTimesAreRegionTimesNel, proximityGroupingSubsetNel, proximityGroupingSupersetNel)
            .tupled
            .map(_ => ImagingRoundsConfiguration(sequence, locusGrouping, proximityFilterStrategy, tracingExclusions))
            .toEither
    }

    private def mkStringTimepoints = (_: Set[Timepoint]).toList.sorted.map(_.get).mkString(", ")
    

    /**
      * Read the configuration of imaging rounds for the experiment, including regional grouping and 
      * exclusions from tracing.
      *
      * @param jsonFile Path to the file from which to parse the configuration
      * @return Either a [[scala.util.Left]]-wrapped nonempty collection of error messages, or 
      *     a [[scala.util.Right]]-wrapped, successfully parsed configuration instance
      */
    def fromJsonFile(jsonFile: os.Path): ErrMsgsOr[ImagingRoundsConfiguration] = 
        Try{ readJsonFile[ujson.Value](jsonFile) }
            .toEither
            .leftMap(e => NonEmptyList.one(e.getMessage))
            .flatMap(fromJson)
    
    /** Try to read a configuration directly from JSON. */
    def fromJson(fullJsonData: ujson.Value): ErrMsgsOr[ImagingRoundsConfiguration] = 
        safeReadAs[Map[String, ujson.Value]](fullJsonData).leftMap(NonEmptyList.one).flatMap(fromJsonMap)

    private def safeReadImagingSequence(json: ujson.Value)(using Reader[ImagingRound]): ErrMsgsOr[ImagingSequence] = 
        safeReadAs[List[ImagingRound]](json)
            .leftMap(NonEmptyList.one)
            .flatMap(ImagingSequence.fromRounds)

    /** JSON codec for an imaging sequence as would be used in a configuration */
    def rwForImagingSequence(using ReadWriter[ImagingRound]): ReadWriter[ImagingSequence] = readwriter[ujson.Value].bimap(
        seq => write(seq.allRounds.toList, indent=2),
        json => safeReadImagingSequence(json).fold(messages => throw new ImagingSequence.DecodingError(messages, json), identity)
    )

    /** Attempt to parse a configuration from a key-value mapping from section name to JSON value. */
    def fromJsonMap(data: Map[String, ujson.Value]): ErrMsgsOr[ImagingRoundsConfiguration] = {
        given rwForRound: Reader[ImagingRound] = ImagingRound.rwForImagingRound
        val roundsNel: ValidatedNel[String, ImagingSequence] = 
            data.get("imagingRounds")
                .toRight(NonEmptyList.one("Missing imagingRounds key!"))
                .flatMap(safeReadImagingSequence)
                .toValidated
        val crudeLocusGroupingNel: ValidatedNel[String, Set[LocusGroup]] = 
            data.get("locusGrouping") match {
                case None | Some(ujson.Null) => Set().validNel
                case Some(fullJson) => safeReadAs[Map[Int, List[Int]]](fullJson) match {
                    case Left(errMsg) => errMsg.invalidNel
                    case Right(maybeGrouping) => maybeGrouping.toList.toNel match {
                        case None => Set().validNel
                        case Some(grouping) => grouping.traverse{ (regionTimeRaw, lociTimesRaw) => 
                            (for {
                                regTime <- Timepoint.fromInt(regionTimeRaw).leftMap("Bad region time as key in locus group! " ++ _)
                                maybeLociTimes <- lociTimesRaw.traverse(Timepoint.fromInt).leftMap("Bad locus time(s) in locus group! " ++ _)
                                lociTimes <- maybeLociTimes.toNel.toRight(s"Empty locus times for region time $regionTimeRaw!").map(_.toNes)
                                _ <- (
                                    if lociTimes.contains(regTime) 
                                    then s"Regional time ${regTime.get} is contained in its own locus times group!".asLeft 
                                    else ().asRight
                                )
                            } yield LocusGroup(regTime, lociTimes)).toValidatedNel
                        }.map(_.toNes.toSortedSet)
                    }
                }
            }
        val proximityFilterStrategyNel: ValidatedNel[String, ProximityFilterStrategy] = {
            data.get("proximityFilterStrategy") match {
                case None => "Missing proximityFilterStrategy section!".invalidNel
                case Some(fullJson) => safeReadAs[Map[String, ujson.Value]](fullJson) match {
                    case Left(message) => message.invalidNel
                    case Right(currentSection) => 
                        val thresholdNel: ValidatedNel[String, Option[PositiveReal]] = 
                            currentSection.get("minimumPixelLikeSeparation") match {
                                case None => None.validNel
                                case Some(json) => safeReadAs[Double](json).flatMap(PositiveReal.either).map(_.some).toValidatedNel
                            }
                        val groupsNel: ValidatedNel[String, Option[NonEmptyList[NonEmptySet[Timepoint]]]] = 
                            currentSection.get("groups") match {
                                case None => None.validNel
                                case Some(subdata) => safeReadAs[List[List[Int]]](subdata)
                                    .flatMap(_.traverse(_.traverse(Timepoint.fromInt))) // Lift all ints to timepoints.
                                    .flatMap(liftToNel(_, "regional grouping".some)) // Entire collection must be nonempty.
                                    .flatMap(_.traverse(liftToNes(_, "regional group".some))) // Each group must be nonempty.
                                    .map(_.some)
                                    .toValidatedNel
                            }
                        currentSection.get("semantic") match {
                            case None => "Missing semantic for proximity filter config section!".invalidNel
                            case Some(s) => safeReadAs[String](s) match {
                                case Left(message) => s"Illegal type of value for regional grouping semantic! Message: $message".invalidNel
                                case Right("UniversalProximityPermission") => (thresholdNel, groupsNel).tupled match {
                                    case fail@Invalid(_) => fail
                                    case Valid((None, None)) => UniversalProximityPermission.validNel
                                    case Valid(_) => "For universal proximity permission, both threshold and groups must be absent.".invalidNel
                                }
                                case Right("UniversalProximityProhibition") => (thresholdNel, groupsNel).tupled match {
                                    case fail@Invalid(_) => fail
                                    case Valid((Some(t), None)) => UniversalProximityProhibition(t).validNel
                                    case Valid(_) => "For universal proximity prohibition, threshold must be present and groups must be absent.".invalidNel
                                }
                                case Right("SelectiveProximityPermission") => (thresholdNel, groupsNel).tupled match {
                                    case fail@Invalid(_) => fail
                                    case Valid((Some(t), Some(g))) => SelectiveProximityPermission(t, g).validNel
                                    case Valid(_) => "For selective proximity permission, threshold and grouping must be present.".invalidNel
                                }
                                case Right("SelectiveProximityProhibition") => (thresholdNel, groupsNel).tupled match {
                                    case fail@Invalid(_) => fail
                                    case Valid((Some(t), Some(g))) => SelectiveProximityProhibition(t, g).validNel
                                    case Valid(_) => "For selective proximity prohibition, threshold and grouping must be present.".invalidNel
                                }
                                case Right(semantic) => 
                                    val errMsg = s"Illegal value for proximity filter semantic: $semantic"
                                    Invalid{ (thresholdNel, groupsNel).tupled.fold(es => NonEmptyList(errMsg, es.toList), _ => NonEmptyList.one(errMsg)) }
                            }
                        }
                }
            }
        }
        val tracingExclusionsNel: ValidatedNel[String, Set[Timepoint]] = 
            data.get("tracingExclusions") match {
                case None | Some(ujson.Null) => Validated.Valid(Set())
                case Some(json) => safeReadAs[List[Int]](json)
                    .flatMap(_.traverse(Timepoint.fromInt))
                    .map(_.toSet)
                    .toValidatedNel
            }
        val checkLocusTimepointCoveringNel: ValidatedNel[String, Boolean] = 
            data.get("checkLocusTimepointCovering") match {
                case None | Some(ujson.Null) => Validated.Valid(true)
                case Some(json) => safeReadAs[Boolean](json).toValidatedNel
            }
        (roundsNel, crudeLocusGroupingNel, proximityFilterStrategyNel, tracingExclusionsNel, checkLocusTimepointCoveringNel).tupled.toEither.flatMap{ 
            case (sequence, crudeLocusGroups, proximityFilterStrategy, exclusions, checkLocusTimepointCovering) =>
                build(sequence, crudeLocusGroups, proximityFilterStrategy, exclusions, checkLocusTimepointCovering)
        }
    }

    /**
     * Create instance, throw exception if any failure occurs
     * 
     * @see [[ImagingRoundsConfiguration.build]]
     */
    def unsafe(
        sequence: ImagingSequence, 
        locusGrouping: Set[LocusGroup], 
        proximityFilterStrategy: ProximityFilterStrategy, 
        tracingExclusions: Set[Timepoint],
        checkLocusTimepointCoveringNel: Boolean,
        ): ImagingRoundsConfiguration = 
        build(sequence, locusGrouping, proximityFilterStrategy, tracingExclusions, checkLocusTimepointCoveringNel).fold(messages => throw new BuildError.FromPure(messages), identity)

    /**
      * Build a configuration instance from JSON data on disk.
      *
      * @param jsonFile Path to file to parse
      * @return Configuration instance
      */
    def unsafeFromJsonFile(jsonFile: os.Path): ImagingRoundsConfiguration = 
        fromJsonFile(jsonFile).fold(messages => throw BuildError.FromJsonFile(messages, jsonFile), identity)

    /**
     * A collection of locus imaging timepoints associated with a single regional imaging timepoint
     * 
     * @param regionalTimepoint The imaging timepoint at which all the DNA loci targeted at the given locus timepoints will all be lit up together
     * @param locusTimepoints The imaging timepoints of DNA loci which will be illuminated together at the given regional timepoint
     */
    private[looptrace] final case class LocusGroup private[looptrace](regionalTimepoint: Timepoint, locusTimepoints: NonEmptySet[Timepoint]):
        require(!locusTimepoints.contains(regionalTimepoint), s"Regional time (${regionalTimepoint.get}) must not be in locus times!")
    
    /** Helpers for working with locus groups */
    object LocusGroup:
        /** Locus groups are ordered by regional timepoint and then by locus timepoints. */
        given orderForLocusGroup: Order[LocusGroup] = Order.by{ case LocusGroup(regionalTimepoint, locusTimepoints) => regionalTimepoint -> locusTimepoints }
    end LocusGroup

    /** Helpers for working with timepoint groupings */
    object RegionalImageRoundGroup:
        /** JSON codec for group of imaging timepoints */
        given rwForRegionalImageRoundGroup: ReadWriter[NonEmptySet[Timepoint]] = readwriter[ujson.Value].bimap(
            group => ujson.Arr(group.toList.map(name => ujson.Num(name.get))*), 
            json => json.arr
                .toList
                .toNel
                .toRight("Empty collection can't parse as group of regional imaging rounds!")
                .flatMap(_.traverse(_.safeInt.flatMap(Timepoint.fromInt)))
                .flatMap(ts => liftToNes(ts.toList))
                .fold(
                    repeats => throw new ujson.Value.InvalidData(json, s"Repeat values for group of regional imaging rounds: $repeats"), 
                    identity
                    )
        )
    end RegionalImageRoundGroup

    /** A way to filter spots if they're too close together */
    sealed trait ProximityFilterStrategy
    
    /** A non-no-op case for exclusion of spots that are too close */
    sealed trait NontrivialProximityFilter extends ProximityFilterStrategy:
        def minSpotSeparation: PositiveReal
    
    /** An exclusion strategy for spots too close that defines which ones are to be considered too close */
    sealed trait SelectiveProximityFilter:
        def grouping: NonEmptyList[NonEmptySet[Timepoint]]
    
    /** "No-op" case for spot filtration--all spots are allowed to occur close together. */
    case object UniversalProximityPermission extends ProximityFilterStrategy
    
    /** Any spot may be deemed "too close" to any other spot. */
    final case class UniversalProximityProhibition(minSpotSeparation: PositiveReal) extends NontrivialProximityFilter
    
    /** Allow spots from timepoints grouped together to violate given separation threshold. */
    final case class SelectiveProximityPermission(
        minSpotSeparation: PositiveReal,
        grouping: NonEmptyList[NonEmptySet[Timepoint]],
        ) extends NontrivialProximityFilter with SelectiveProximityFilter
    
    /** Forbid spots from timepoints grouped together to violate given separation threshold. */
    final case class SelectiveProximityProhibition(
        minSpotSeparation: PositiveReal, 
        grouping: NonEmptyList[NonEmptySet[Timepoint]],
        ) extends NontrivialProximityFilter with SelectiveProximityFilter

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
                s"$total total - $unique unique = ${total - unique} repeated items" ++ context.fold("")(ctx => s" for $ctx"), 
                candidate
                )
        )
end ImagingRoundsConfiguration