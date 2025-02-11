package at.ac.oeaw.imba.gerlich.looptrace

import scala.collection.immutable.SortedSet
import scala.language.adhocExtensions // to extend ujson.Value.InvalidData
import scala.util.Try
import scala.util.chaining.*
import cats.*
import cats.data.{ EitherNel, NonEmptyList, NonEmptyMap, NonEmptySet, Validated, ValidatedNel }
import cats.data.Validated.{ Invalid, Valid }
import cats.syntax.all.*
import mouse.boolean.*
import ujson.IncompleteParseException
import upickle.default.*
import com.typesafe.scalalogging.LazyLogging

import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2
import at.ac.oeaw.imba.gerlich.gerlib.geometry.{ DistanceThreshold, PiecewiseDistance }
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.imagingTimepoint.given
import at.ac.oeaw.imba.gerlich.gerlib.json.syntax.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.LocusGroup
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.{ readJsonFile, safeReadAs }
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanDistance

/** Typical looptrace declaration/configuration of imaging rounds and how to use them */
final case class ImagingRoundsConfiguration private(
    sequence: ImagingSequence, 
    locusGrouping: Set[ImagingRoundsConfiguration.LocusGroup], // should be empty iff there are no locus rounds in the sequence
    proximityFilterStrategy: ImagingRoundsConfiguration.ProximityFilterStrategy,
    private val maybeMergeRulesForTracing: Option[(NonEmptyList[ImagingRoundsConfiguration.TraceIdDefinitionRule], Boolean)],
    // TODO: We could, by default, skip regional and blank imaging rounds (but do use repeats).
    tracingExclusions: Set[ImagingTimepoint], // Timepoints of imaging rounds to not use for tracing
):
    import ImagingRoundsConfiguration.given
    
    /** Simply take the rounds from the contained imagingRounds sequence. */
    final def allRounds: NonEmptyList[ImagingRound] = sequence.allRounds
        
    final def mergeRules: Option[NonEmptyList[ImagingRoundsConfiguration.TraceIdDefinitionRule]] = 
        maybeMergeRulesForTracing.map(_._1)
    
    final def discardRoisNotInGroupsOfInterest: Boolean = maybeMergeRulesForTracing.fold(false)(_._2)

    /** The number of imaging rounds is the length of the imagingRounds sequence. */
    final def numberOfRounds: Int = sequence.length
    
    /** Faciliate lookup of reindexed timepoint for visualisation. */
    lazy val lookupReindexedImagingTimepoint: Map[ImagingTimepoint, Map[ImagingTimepoint, Int]] = {
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
    private given Ordering[ImagingTimepoint] = summon[Order[ImagingTimepoint]].toOrdering
    
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
    def checkTimesSubset(knownTimes: Set[ImagingTimepoint])(times: Set[ImagingTimepoint], context: String): ValidatedNel[String, Unit] = 
        (times -- knownTimes).toList match {
            case Nil => ().validNel
            case unknown => s"Unknown timepoint(s) ($context): ${unknown.sorted.map(_.show_).mkString(", ")}".invalidNel
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
      * @param maybeMergeRules How to merge ROIs from different timepoints
      * @param checkLocusTimepointCovering Whether to validate that the union of the locusGrouping section values covers all locus imaging timepoints
      * @return Either a [[scala.util.Left]]-wrapped nonempty list of error messages, or a [[scala.util.Right]]-wrapped built instance
      */
    def build(
        sequence: ImagingSequence, 
        locusGrouping: Set[LocusGroup], 
        proximityFilterStrategy: ProximityFilterStrategy, 
        tracingExclusions: Set[ImagingTimepoint], 
        maybeMergeRules: Option[(NonEmptyList[TraceIdDefinitionRule], Boolean)],
        checkLocusTimepointCovering: Boolean,
    ): ErrMsgsOr[ImagingRoundsConfiguration] = {
        val knownTimes = sequence.allTimepoints
        // Regardless of the subtype of proximityFilterStrategy, we need to check that any tracing exclusion timepoint is a known timepoint.
        val tracingSubsetNel = checkTimesSubset(knownTimes.toSortedSet)(tracingExclusions, "tracing exclusions")
        // TODO: consider checking that every regional timepoint in the sequence is represented in the locusGrouping.
        // See: https://github.com/gerlichlab/looptrace/issues/270
        val uniqueTimepointsInLocusGrouping = locusGrouping.map(_.locusTimepoints).foldLeft(Set.empty[ImagingTimepoint])(_ ++ _.toSortedSet)
        val (locusTimeSubsetNel, locusTimeSupersetNel) = {
            if locusGrouping.isEmpty then (().validNel, ().validNel) else {
                val locusTimesInSequence = sequence.locusRounds.map(_.time).toSet
                // First, check that the union of values in the locus grouping is a subset of the locus-specific imaging rounds.
                val subsetNel = (uniqueTimepointsInLocusGrouping -- locusTimesInSequence).toList match {
                    case Nil => ().validNel
                    case ts => s"${ts.length} timepoint(s) in locus grouping and not found as locus imaging timepoints: ${mkStringTimepoints(ts)}".invalidNel
                }
                // Then, check that each locus-specific imaging round is in the locus grouping, as appropriate..
                val supersetNel = 
                    if checkLocusTimepointCovering
                    then
                        val missing = locusTimesInSequence -- uniqueTimepointsInLocusGrouping
                        logger.debug(s"${missing.size} locus imaging timepoint(s) missing from the locus grouping: ${mkStringTimepoints(missing.toList)}")
                        val (correctlyMissing, wronglyMissing) = missing.partition(tracingExclusions.contains)
                        logger.debug(s"${correctlyMissing.size} locus imaging timepoint(s) CORRECTLY missing from locus grouping: ${mkStringTimepoints(correctlyMissing.toList)}")
                        if wronglyMissing.isEmpty 
                        then ().validNel
                        else s"${wronglyMissing.size} locus timepoint(s) in imagingRounds and not found in locusGrouping (nor in tracingExclusions): ${mkStringTimepoints(wronglyMissing.toList)}".invalidNel
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
        val uniqueRegionalTimes = sequence.regionRounds.map(_.time).toList.toSet
        
        /* Every timepoint in a proximity grouping must be a regional (rather than locus-specific) timepoint. */
        val (proximityGroupingSubsetNel, proximityGroupingSupersetNel) = proximityFilterStrategy match {
            case (UniversalProximityPermission | UniversalProximityProhibition(_)) => 
                // In the trivial case, we have no more validation work to do.
                (().validNel, ().validNel)
            case s: (SelectiveProximityPermission | SelectiveProximityProhibition) => 
                // In the nontrivial case, check for set equivalance of timepoints b/w imaging sequence and grouping.
                val uniqueGroupedTimes = s.grouping.reduce(_ ++ _).toList.toSet
                val subsetNel = checkTimesSubset(uniqueRegionalTimes)(uniqueGroupedTimes, "proximity filter's grouping (rel. to regionals in imaging sequence)")
                val supersetNel = checkTimesSubset(uniqueGroupedTimes)(uniqueRegionalTimes, "regionals in imaging sequence (rel. to proximity filter strategy)")
                (subsetNel, supersetNel)
        }

        /* Every timepoint ID to merge for tracing must a regional (rather than locus-specific) timepoint. */
        val (idsToMergeAreAllRegionalNel, noRepeatsInLocusTimeSetsForRegionalTimesToMerge) = 
            import at.ac.oeaw.imba.gerlich.gerlib.collections.given // SemigroupK[AtLeast2[Set, *]]
            maybeMergeRules match {
                case None => (().validNel, ().validNel)
                case Some((rules, _)) => 
                    val allAreRegional: ValidatedNel[String, Unit] = 
                        import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2.syntax.toSet
                        (rules.map(_.mergeGroup.members).reduceK.toSet -- uniqueRegionalTimes)
                            .toList
                            .sorted
                            .toNel
                            .toLeft(())
                            .leftMap(nonRegionalTimesInRules => 
                                s"${nonRegionalTimesInRules.size} non-regional time(s) in merge rules: ${nonRegionalTimesInRules.toList.sorted.map(_.show_).mkString(", ")}"
                            )
                            .toValidatedNel
                    
                    val noRepeatsInLocusTimesOfRegionalsToMerge: ValidatedNel[String, Unit] = 
                        import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2.syntax.toList
                        
                        // First, build the lookup of locus times by regional time.
                        val locusTimesByRegional = locusGrouping
                            .foldLeft(Map.empty[ImagingTimepoint, NonEmptySet[ImagingTimepoint]]){ (acc, g) => 
                                val rt = g.regionalTimepoint
                                val lts = g.locusTimepoints
                                acc + (rt -> acc.get(rt).fold(lts)(lts ++ _))
                            }
                        
                        // Then, find repeated locus timepoints WITHIN each merge group.
                        type GroupId = Int
                        val repeatsByGroup: Map[GroupId, NonEmptyMap[ImagingTimepoint, AtLeast2[Set, ImagingTimepoint]]] = 
                            def processOneRule = (r: TraceIdDefinitionRule) => 
                                r.mergeGroup.members.toList.foldRight(Map.empty[ImagingTimepoint, NonEmptySet[ImagingTimepoint]]){
                                    (rt, acc) => locusTimesByRegional
                                        .get(rt)
                                        .fold(acc)(_
                                            .toList
                                            .foldRight(acc){ (lt, m) => 
                                                m + (lt -> m.get(lt).fold(NonEmptySet.one(rt))(_.add(rt))) 
                                            }
                                        )
                                }
                            
                            // Determine if a single rule's inverse mapping from locus timepoints to regional timepoints is problematic.
                            def getBadResult: Map[ImagingTimepoint, NonEmptySet[ImagingTimepoint]] => Option[NonEmptyMap[ImagingTimepoint, AtLeast2[Set, ImagingTimepoint]]] = _
                                .view
                                .mapValues(_.toSortedSet.toSet.pipe(AtLeast2.either).pipe(_.toOption))
                                .flatMap{ (locTime, maybeRegTimes) => maybeRegTimes.map(locTime -> _) }
                                .pipe(scala.collection.immutable.SortedMap.from)
                                .pipe(NonEmptyMap.fromMap)
                                
                            rules.zipWithIndex.toList.flatMap{ (r, i) => getBadResult(processOneRule(r)).map(i -> _) }.toMap
                        
                        // If no repeats, we're all good; otherwise, make an error message.
                        if repeatsByGroup.isEmpty 
                        then ().validNel
                        else s"Regionals timepoints to merge for tracing map to overlapping locus timepoint sets; here are the repeat(s): $repeatsByGroup".invalidNel
                    
                    (allAreRegional, noRepeatsInLocusTimesOfRegionalsToMerge)
            }

        (
            tracingSubsetNel, 
            locusTimeSubsetNel, 
            locusTimeSupersetNel, 
            locusGroupTimesAreRegionTimesNel, 
            proximityGroupingSubsetNel, 
            proximityGroupingSupersetNel, 
            idsToMergeAreAllRegionalNel,
            noRepeatsInLocusTimeSetsForRegionalTimesToMerge,
        )
            .tupled
            // We ignore the acutal values (Unit) because this was just to accumulate errors.
            .map(_ => ImagingRoundsConfiguration(sequence, locusGrouping, proximityFilterStrategy, maybeMergeRules, tracingExclusions))
            .toEither
    }

    private def mkStringTimepoints = (_: List[ImagingTimepoint]).sorted.map(_.show_).mkString(", ")

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
                                regTime <- ImagingTimepoint.fromInt(regionTimeRaw).leftMap("Bad region time as key in locus group! " ++ _)
                                maybeLociTimes <- lociTimesRaw.traverse(ImagingTimepoint.fromInt).leftMap("Bad locus time(s) in locus group! " ++ _)
                                lociTimes <- maybeLociTimes.toNel.toRight(s"Empty locus times for region time $regionTimeRaw!").map(_.toNes)
                                _ <- (
                                    if lociTimes.contains(regTime) 
                                    then s"Regional time ${regTime.show_} is contained in its own locus times group!".asLeft 
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
                        val thresholdNel: ValidatedNel[String, Option[NonnegativeReal]] = 
                            currentSection.get("minimumPixelLikeSeparation") match {
                                case None => None.validNel
                                case Some(json) => safeReadAs[Double](json).flatMap(NonnegativeReal.either).map(_.some).toValidatedNel
                            }
                        val groupsNel: ValidatedNel[String, Option[NonEmptyList[NonEmptySet[ImagingTimepoint]]]] = 
                            currentSection.get("groups") match {
                                case None => None.validNel
                                case Some(subdata) => safeReadAs[List[List[Int]]](subdata)
                                    .flatMap(_.traverse(_.traverse(ImagingTimepoint.fromInt))) // Lift all ints to timepoints.
                                    .flatMap(liftToNel(_, "regional grouping".some)) // Entire collection must be nonempty.
                                    .flatMap(_.traverse(liftToNes(_, "regional group".some))) // Each group must be nonempty.
                                    .map(_.some)
                                    .toValidatedNel
                            }
                        currentSection.get("semantic") match {
                            case None => "Missing semantic for proximity filter config section!".invalidNel
                            case Some(s) => (safeReadAs[String](s), (thresholdNel, groupsNel).tupled) match {
                                /* First, the 2 obvious fail cases */
                                case (Left(message), _) => s"Illegal type of value for regional grouping semantic! Message: $message".invalidNel
                                case (_, fail@Invalid(_)) => fail
                                /* Then, the good cases */
                                case (Right("UniversalProximityPermission"), Valid((None, None))) => UniversalProximityPermission.validNel
                                case (Right("UniversalProximityPermission"), Valid(_)) => 
                                    "For universal proximity permission, both threshold and groups must be absent.".invalidNel
                                case (Right("UniversalProximityProhibition"), Valid((Some(t), None))) => UniversalProximityProhibition(t).validNel
                                case (Right("UniversalProximityProhibition"), Valid(_)) => 
                                    "For universal proximity prohibition, threshold must be present and groups must be absent.".invalidNel
                                case (Right("SelectiveProximityPermission"), Valid((Some(t), Some(g)))) => SelectiveProximityPermission(t, g).validNel
                                case (Right("SelectiveProximityPermission"), Valid(_)) => 
                                    "For selective proximity permission, threshold and grouping must be present.".invalidNel
                                case (Right("SelectiveProximityProhibition"), Valid((Some(t), Some(g)))) => SelectiveProximityProhibition(t, g).validNel
                                case (Right("SelectiveProximityProhibition"), Valid(_)) => 
                                    "For selective proximity prohibition, threshold and grouping must be present.".invalidNel
                                /* Finally, the illegal semantic case */
                                case (Right(semantic), _) => 
                                    val errMsg = s"Illegal value for proximity filter semantic: $semantic"
                                    Invalid{ (thresholdNel, groupsNel).tupled.fold(es => NonEmptyList(errMsg, es.toList), _ => NonEmptyList.one(errMsg)) }
                            }
                        }
                }
            }
        }
        val tracingExclusionsNel: ValidatedNel[String, Set[ImagingTimepoint]] = 
            data.get("tracingExclusions") match {
                case None | Some(ujson.Null) => Validated.Valid(Set())
                case Some(json) => safeReadAs[List[Int]](json)
                    .flatMap(_.traverse(ImagingTimepoint.fromInt))
                    .map(_.toSet)
                    .toValidatedNel
            }
        val maybeMergeRulesNel: ValidatedNel[String, Option[(NonEmptyList[TraceIdDefinitionRule], Boolean)]] = 
            val sectionKey = "mergeRulesForTracing"
            data.get(sectionKey) match {
                case None | Some(ujson.Null) => 
                    logger.debug(s"No section '$sectionKey', ignoring")
                    Validated.Valid(None)
                case Some(jsonData) => 
                    TraceIdDefinitionRulesSet.fromJson(jsonData).toValidated.map(_.some)
            }
        val checkLocusTimepointCoveringNel: ValidatedNel[String, Boolean] = 
            data.get("checkLocusTimepointCovering") match {
                case None | Some(ujson.Null) => Validated.Valid(true)
                case Some(json) => safeReadAs[Boolean](json).toValidatedNel
            }
        (roundsNel, crudeLocusGroupingNel, proximityFilterStrategyNel, tracingExclusionsNel, maybeMergeRulesNel, checkLocusTimepointCoveringNel)
            .tupled
            .toEither
            .flatMap{ case (sequence, crudeLocusGroups, proximityFilterStrategy, exclusions, maybeMergeRules, checkLocusTimepointCovering) =>
                build(sequence, crudeLocusGroups, proximityFilterStrategy, exclusions, maybeMergeRules, checkLocusTimepointCovering)
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
        tracingExclusions: Set[ImagingTimepoint],
        maybeMergeRules: Option[(NonEmptyList[TraceIdDefinitionRule], Boolean)],
        checkLocusTimepointCoveringNel: Boolean,
    ): ImagingRoundsConfiguration = 
        build(
            sequence, 
            locusGrouping, 
            proximityFilterStrategy, 
            tracingExclusions, 
            maybeMergeRules, 
            checkLocusTimepointCoveringNel,
        ).fold(
            messages => throw new BuildError.FromPure(messages), 
            identity,
        )

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
    private[looptrace] final case class LocusGroup private[looptrace](regionalTimepoint: ImagingTimepoint, locusTimepoints: NonEmptySet[ImagingTimepoint]):
        require(!locusTimepoints.contains(regionalTimepoint), s"Regional time (${regionalTimepoint.show_}) must not be in locus times!")
    
    /** Helpers for working with locus groups */
    object LocusGroup:
        /** Locus groups are ordered by regional timepoint and then by locus timepoints. */
        given orderForLocusGroup: Order[LocusGroup] = Order.by{ case LocusGroup(regionalTimepoint, locusTimepoints) => regionalTimepoint -> locusTimepoints }
    end LocusGroup

    /** Helpers for working with timepoint groupings */
    object RegionalImageRoundGroup:
        /** JSON codec for group of imaging timepoints */
        given rwForRegionalImageRoundGroup: ReadWriter[NonEmptySet[ImagingTimepoint]] = readwriter[ujson.Value].bimap(
            group => ujson.Arr(group.toList.map(_.asJson)*), 
            json => json.arr
                .toList
                .toNel
                .toRight("Empty collection can't parse as group of regional imaging rounds!")
                .flatMap(_.traverse(_.safeInt.flatMap(ImagingTimepoint.fromInt)))
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
        def minSpotSeparation: NonnegativeReal
    
    /** An exclusion strategy for spots too close that defines which ones are to be considered too close */
    sealed trait SelectiveProximityFilter:
        def grouping: NonEmptyList[NonEmptySet[ImagingTimepoint]]
    
    /** "No-op" case for spot filtration--all spots are allowed to occur close together. */
    case object UniversalProximityPermission extends ProximityFilterStrategy
    
    /** Any spot may be deemed "too close" to any other spot. */
    final case class UniversalProximityProhibition(minSpotSeparation: NonnegativeReal) extends NontrivialProximityFilter
    
    /** Allow spots from timepoints grouped together to violate given separation threshold. */
    final case class SelectiveProximityPermission(
        minSpotSeparation: NonnegativeReal,
        grouping: NonEmptyList[NonEmptySet[ImagingTimepoint]],
        ) extends NontrivialProximityFilter with SelectiveProximityFilter
    
    /** Forbid spots from timepoints grouped together to violate given separation threshold. */
    final case class SelectiveProximityProhibition(
        minSpotSeparation: NonnegativeReal, 
        grouping: NonEmptyList[NonEmptySet[ImagingTimepoint]],
        ) extends NontrivialProximityFilter with SelectiveProximityFilter

    /** How to filter/discard ROIs on the basis of occurrence in proximity to other group members */
    enum RoiPartnersRequirementType:
        /** Require proximity of 'at least one' group member (i.e., logical OR). */
        case Disjunctive
        /** Require proximity of 'all' group members (i.e., logical AND). */
        case Conjunctive
        /** Group ROIs for tracing if they're proximal, but do 'not' discard singletons from the groups. */
        case Lackadaisical

    /** A grouping of elements {@code E} to consider for proximity relative to some distance threshold */
    final case class ProximityGroup[T <: DistanceThreshold, E](
        distanceThreshold: T, 
        members: AtLeast2[Set, E]
    )

    /** Helpers for working with the proximity group data type */
    object ProximityGroup:
        /** The key for the proximity-defining distance threshold value */
        private[ImagingRoundsConfiguration] val thresholdKey = "distanceThreshold"
        
        private[ImagingRoundsConfiguration] val requirementTypeKey = "requirementType"
    end ProximityGroup

    /** How to redefine trace IDs and filter ROIs on the basis of proximity to one another */
    final case class TraceIdDefinitionRule(
        name: TraceGroupId,
        mergeGroup: ProximityGroup[EuclideanDistance.Threshold, ImagingTimepoint],
        requirement: RoiPartnersRequirementType, 
    )

    /** Helpers for working with the data type for trace ID definition and filtration */
    object TraceIdDefinitionRulesSet:
        /** The key which maps to the collection of groups */
        private val groupsKey = "groups"
        private val strictnessKey = "discardRoisNotInGroupsOfInterest"

        def fromJson(json: ujson.Readable): EitherNel[String, (NonEmptyList[TraceIdDefinitionRule], Boolean)] = 
            Try{ read[Map[String, ujson.Value]](json) }
                .toEither
                .leftMap(e => NonEmptyList.one(s"Cannot read JSON as key-value pairs: ${e.getMessage}"))
                .flatMap{ kvPairs => 
                    val filtrationStrictnessNel = kvPairs.get(strictnessKey) match {
                        case None | Some(ujson.Null) => s"Missing key: '$strictnessKey'".invalidNel
                        case Some(v) => UJsonHelpers.safeReadAs[Boolean](v).toValidatedNel
                    }
                    val maybeThresholdNel = kvPairs.get(ProximityGroup.thresholdKey) match {
                        case None => 
                            None.validNel[String]
                        case Some(v) => UJsonHelpers.safeReadAs[Double](v)
                            .flatMap(NonnegativeReal.either)
                            .map(d => EuclideanDistance.Threshold(d).some)
                            .toValidatedNel
                    }
                    val maybeRequirementTypeNel = kvPairs.get(ProximityGroup.requirementTypeKey) match {
                        case None => 
                            None.validNel[String]
                        case Some(v) => UJsonHelpers.safeReadAs[String](v)
                            .flatMap{ s => Try{ 
                                    RoiPartnersRequirementType.valueOf(s) 
                                }
                                .fold(
                                    Function.const{ s"Can't parse value for '${ProximityGroup.requirementTypeKey}': $s".asLeft },
                                    _.some.asRight,
                                )
                            }
                            .toValidatedNel
                    }
                    (filtrationStrictnessNel, maybeThresholdNel, maybeRequirementTypeNel)
                        .tupled
                        .toEither
                        .flatMap{ (strictness, maybeThreshold, maybeRequirementType) => 
                            kvPairs.get(groupsKey)
                                .toRight(s"Missing key for groups: $groupsKey")
                                .flatMap(_.objOpt.toRight(s"Can't parse groups (from '$groupsKey') as map-like"))
                                .flatMap(_.toList.toNel.toRight(s"Data for groups (from '$groupsKey') is empty"))
                                .leftMap(NonEmptyList.one)
                                .flatMap(_.nonEmptyTraverse{ (k, v) => parseGroupMembersSimple(v).bimap(NonEmptyList.one, k -> _) })
                                .flatMap(groups => (maybeThreshold, maybeRequirementType) match {
                                    case (None, None) => NonEmptyList.one("Missing threshold and requirement type for ROI merge").asLeft
                                    case (Some(threshold), None) => NonEmptyList.one("Missing requirement type for ROI merge").asLeft
                                    case (None, Some(reqType)) => NonEmptyList.one("Missing threshold for ROI merge").asLeft
                                    case (Some(threshold), Some(reqType)) => 
                                        groups.traverse{ (nameCandidate, group) => 
                                            TraceGroupId.fromString(nameCandidate).bimap(
                                                NonEmptyList.one, 
                                                name => TraceIdDefinitionRule(
                                                    name,
                                                    ProximityGroup(threshold, group), 
                                                    reqType,
                                                )
                                            )
                                        }
                                })
                                .flatMap{ rules => 
                                    import AtLeast2.syntax.toList
                                    rules.toList
                                        .flatMap(_.mergeGroup.members.toList)
                                        .groupBy(identity)
                                        .view
                                        .mapValues(_.size)
                                        .toList
                                        .filter(_._2 > 1) match {
                                            case Nil => rules.asRight
                                            case repeats => 
                                                NonEmptyList.one(s"${repeats.size} repeated item(s) in merge rules: ${repeats}").asLeft 
                                        }
                                }
                                .map(_ -> strictness)
                        }
                }
        
        private def parseThreshold: ujson.Value => Either[String, EuclideanDistance.Threshold] = v =>
            v.numOpt
                .toRight(s"Value to decoder as a distance threshold ins't a number: $v")
                .flatMap(NonnegativeReal.either)
                .map(EuclideanDistance.Threshold.apply)

        // Use this to parse the actual collection of ImagingTimepoint values.
        private def parseGroupMembersSimple(json: ujson.Value): Either[String, AtLeast2[Set, ImagingTimepoint]] = 
            import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2.syntax.*
            import at.ac.oeaw.imba.gerlich.gerlib.json.instances.collections.given
            given Reader[ImagingTimepoint] = reader[ujson.Value].map(_.int).map(ImagingTimepoint.unsafeLift)
            UJsonHelpers.safeReadAs[AtLeast2[List, ImagingTimepoint]](json).flatMap{ vs =>
                val uniques = vs.toList.toSet
                (uniques.size === vs.size).either(
                    s"${uniques.size} unique value(s), but ${vs.size} total",
                    AtLeast2.unsafe(uniques)
                )
            }
    end TraceIdDefinitionRulesSet

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