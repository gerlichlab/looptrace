package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import scala.util.chaining.*
import cats.*
import cats.data.*
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import cats.syntax.all.*
import fs2.data.csv.*
import io.github.iltotore.iron.:|
import io.github.iltotore.iron.cats.given
import io.github.iltotore.iron.constraint.any.Not
import io.github.iltotore.iron.constraint.collection.*
import io.github.iltotore.iron.constraint.numeric.{Greater, Negative}
import mouse.boolean.*
import scopt.*
import squants.space.{Length, Nanometers}

import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.collections.{AtLeast2, lookupBySubset}
import at.ac.oeaw.imba.gerlich.gerlib.geometry.{Centroid, EuclideanDistance}
import at.ac.oeaw.imba.gerlich.gerlib.graph.{SimplestGraph, buildSimpleGraph}
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.{
  NucleusDesignationColumnName,
  SpotChannelColumnName
}
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{
  readCsvToCaseClasses,
  writeCaseClassesToCsv
}
import at.ac.oeaw.imba.gerlich.gerlib.json.syntax.asJson
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.{
  RoiPartnersRequirementType,
  TraceIdDefinitionRule
}
import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.{
  MergeContributorsColumnNameForAssessedRecord,
  RoiIndexColumnName
}
import at.ac.oeaw.imba.gerlich.looptrace.csv.getCsvRowDecoderForImagingChannel
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.tracing.getCsvRowEncoderForTraceIdAssignmentWithoutRoiIndex
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.IndexedDetectedSpot
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

/** Assign trace IDs to regional spots, considering the potential to group some
  * together for downstream analytical purposes.
  */
object AssignTraceIds extends ScoptCliReaders, StrictLogging:
  val ProgramName = "AssignTraceIds"

  final case class CliConfig(
      roundsConfig: ImagingRoundsConfiguration =
        null, // unconditionally required
      roisFile: os.Path = null, // unconditionally required
      pixels: Pixels3D = null, // required,
      outputFile: os.Path = null, // unconditionally required
      skipsFile: os.Path = null // unconditionally required
  )

  val parserBuilder = OParser.builder[CliConfig]

  given Eq[os.Path] = Eq.fromUniversalEquals

  def main(args: Array[String]): Unit = {
    import parserBuilder.*

    val parser = OParser.sequence(
      programName(ProgramName),
      head(ProgramName, BuildInfo.version),
      opt[ImagingRoundsConfiguration]("configuration")
        .required()
        .action((rounds, cliConf) => cliConf.copy(roundsConfig = rounds))
        .text("Path to file specifying the imaging rounds configuration"),
      opt[os.Path]("roisFile")
        .required()
        .action((f, c) => c.copy(roisFile = f))
        .validate(f =>
          os.isFile(f)
            .either(s"Alleged ROIs file path isn't an extant file: $f", ())
        )
        .text("Path to the file with the ROIs for which to define trace IDs"),
      opt[Pixels3D]("pixels")
        .required()
        .action((ps, c) => c.copy(pixels = ps))
        .text("How many nanometers per unit in each direction (x, y, z)"),
      opt[os.Path]('O', "outputFile")
        .required()
        .action((f, c) => c.copy(outputFile = f))
        .text("Path to file to which to write main output"),
      opt[os.Path]("skipsFile")
        .required()
        .action((f, c) => c.copy(skipsFile = f))
        .text("Path to location to which to write skips"),
      checkConfig { c =>
        if c.roisFile =!= c.outputFile then success
        else failure(s"ROIs file and output file are the same! ${c.roisFile}")
      },
      checkConfig { c =>
        if c.roisFile =!= c.skipsFile then success
        else failure(s"ROIs file and skips file are the same! ${c.roisFile}")
      },
      checkConfig { c =>
        if c.outputFile =!= c.skipsFile then success
        else failure(s"Output file skips file are the same! ${c.outputFile}")
      }
    )

    OParser.parse(parser, args, CliConfig()) match {
      case None =>
        throw new Exception(
          s"Illegal CLI use of '${ProgramName}' program. Check --help"
        ) // CLI parser gives error message.
      case Some(opts) =>
        workflow(
          roundsConfig = opts.roundsConfig,
          roisFile = opts.roisFile,
          pixels = opts.pixels,
          outputFile = opts.outputFile,
          skipsFile = opts.skipsFile
        )
    }
  }

  private def checkTraceId(
      offLimits: NonEmptySet[TraceId]
  )(tid: TraceId): Unit =
    if offLimits contains tid then {
      throw new Exception(
        s"Trace ID is already a ROI index and can't be used: ${tid.show_}"
      )
    }

  /** Define a lookup table for the distance threshold which defines points'
    * pairwise proximity.
    *
    * Specifically, use the merge groups declared in the given rules, to map
    * pairs of imaging timepoints to a distance value. After this mapping's
    * constructed, it will contain all pairs of imaging timepoints (including
    * their reflection, so that lookup won't be sensitive to the permutation of
    * the timepoints in a query tuple) it its set of keys, with each key mapped
    * to a distance between centroids, beneath which a pair of points which
    * comes from the mapped timepoint pair will be considered sufficiently
    * proximal to "merge" into the same tracing structure.
    *
    * @param rules
    *   A configuration's section pertaining to how to merge detected spot ROIs
    *   for tracing a larger structure
    * @return
    *   A mapping from pair of imaging timepoints to a distance threshold value,
    *   in which the value represents the maximal distance by which points from
    *   the given timepoints may be separated and still be considered
    *   sufficiently proximal to merge for tracing
    */
  private def definePairwiseDistanceThresholds(
      rules: NonEmptyList[TraceIdDefinitionRule]
  ): Map[(ImagingTimepoint, ImagingTimepoint), EuclideanDistance] =
    import AtLeast2.syntax.toList
    rules
      .map(_.mergeGroup)
      .map { g =>
        g.members.toList
          .combinations(2)
          .toList
          .flatMap {
            case t1 :: t2 :: Nil =>
              val dt = g.distanceThreshold
              List((t1 -> t2) -> dt, (t2 -> t1) -> dt)
            case ts =>
              throw new Exception(
                s"Got ${ts.length} elements when taking combinations of 2!"
              )
          }
      }
      .foldLeft(Map()) { (thresholds, g) =>
        g.foldLeft(thresholds) { case (acc, (k, v)) =>
          if acc contains k
          then
            throw new Exception(
              s"Key $k is already mapped to a distance threshold!"
            )
          else acc + (k -> v)
        }
      }

  /** Build up the (unweighted, undirected) graph structure in which ROIs are
    * connected iff they're sufficiently proximal.
    *
    * @param rules
    *   The groups of timepoints and distance thresholds which define pairwise
    *   proximity of points (i.e., the regional FISH spots)
    * @param pixels
    *   A definition of the physical units (e.g., nanometers) corresponding to
    *   the image units (pixels) in each dimension of an image
    * @param records
    *   The ROIs from which to build the graph (i.e., the nodes, or vertex set)
    * @return
    *   A graph structure with just the ROI index/IDs as node values, with edges
    *   representing an instance of satisfaction of the proximity criterion
    *   according to the distance threshold corresponding to the pair of imaging
    *   timepoints from whence the ROIs originated
    */
  private def computeNeighborsGraph(
      rules: NonEmptyList[TraceIdDefinitionRule],
      pixels: Pixels3D
  )(records: NonEmptyList[InputRecord]): SimplestGraph[RoiIndex] =
    val lookupProximity: Map[
      (ImagingTimepoint, ImagingTimepoint),
      (InputRecord, InputRecord) => Boolean
    ] =
      definePairwiseDistanceThresholds(rules).view.mapValues {
        threshold => (r1: InputRecord, r2: InputRecord) =>
          import at.ac.oeaw.imba.gerlich.looptrace.syntax.bifunctor.mapBoth
          val (p1, p2) = (r1, r2).mapBoth(_.centroid.asPoint)
          val distance = euclideanDistanceBetweenImagePoints(pixels)(p1, p2)
          distance < threshold
      }.toMap
    val edgeEndpoints: Set[(RoiIndex, RoiIndex)] =
      given Order[FieldOfViewLike]:
        override def compare(x: FieldOfViewLike, y: FieldOfViewLike): Int =
          (x, y) match {
            case (fov1: FieldOfView, fov2: FieldOfView)   => fov1 compare fov2
            case (pos1: PositionName, pos2: PositionName) => pos1 compare pos2
            case _ => throw new Exception(s"Cannot compare $x to $y")
          }
      records
        .groupBy(r =>
          r.context.fieldOfView -> r.context.channel
        ) // Only merge ROIs from the same context (FOV, channel).
        .values // Once records are properly grouped by context, we no longer care about those context keys.
        .flatMap(
          // We do our pairwise calculations only within each group, but then flatten to collect all results.
          _.toList.combinations(2).flatMap { // flatMap here b/c of optionality of output from each record
            case r1 :: r2 :: Nil =>
              // Here, by only getting back a distance threshold value for pairs of timepoints
              // which were in the set of merge rules for tracing, we ensure that edges in
              // the graph are never between ROIs which just happen to be close together but
              // which weren't part of the declared structure for tracing.
              lookupProximity
                // First, these records' timepoints may not have been in the rules set and
                // may therefore not need to be tested for proximity.
                .get(r1.timepoint -> r2.timepoint)
                // Given that this pair ARE jointly in a merge rule, emit a
                // pair of edge endpoints if and only if these records are proximal.
                .flatMap(_(r1, r2).option(r1.index -> r2.index))
            case notPair =>
              throw new Exception(
                s"Got ${notPair.length} element(s) when taking pairs!"
              )
          }
        )
        .toSet
    // Ensure each record gets a node, and add the discovered edges.
    buildSimpleGraph(records.map(_.index).toList.toSet, edgeEndpoints)

  // Start trace IDs with 1 more than max ROI ID/index.
  private def getInitialTraceId(roiIds: NonEmptyList[RoiIndex]): TraceId =
    val maxRoiId = roiIds.toList.max(using Order[RoiIndex].toOrdering)
    TraceId.unsafe(NonnegativeInt(1) + maxRoiId.get)

  private type TimepointExpectationLookup =
    NonEmptyMap[ImagingTimepoint, TraceIdDefinitionRule]

  private[looptrace] def labelRecordsWithTraceId(
      rules: NonEmptyList[TraceIdDefinitionRule],
      discardIfNotInGroupOfInterest: Boolean,
      pixels: Pixels3D
  )(records: NonEmptyList[InputRecord]): List[InputRecordFate] =
    /* Necessary imports and type aliases */
    import AtLeast2.syntax.{toNes, toSet}

    val lookupRecord: NonEmptyMap[RoiIndex, InputRecord] =
      records.map(r => r.index -> r).toNem

    val lookupRule: TimepointExpectationLookup =
      // Provide a way to get the expected group members and requirement stringency for a given timepoint.
      given [V] => Order[(ImagingTimepoint, V)] = Order.by(_._1)
      given semigroup: Semigroup[TimepointExpectationLookup] =
        Semigroup.instance { (x, y) =>
          val collisions = x.keys & y.keys
          if collisions.isEmpty then x ++ y
          else
            throw new Exception(
              s"${collisions.size} key collision(s) between lookups to combine: $collisions"
            )
        }
      rules.reduceMap { r =>
        r.mergeGroup.members.toNes.map(_ -> r).toNonEmptyList.toNem
      }

    val lookupTraceGroupId: NonEmptySet[ImagingTimepoint] => Either[
      AtLeast2[List, (Set[ImagingTimepoint], TraceGroupId)],
      TraceGroupMaybe
    ] =
      val traceGroupIdByRegTimeSet
          : NonEmptyMap[Set[ImagingTimepoint], TraceGroupId] =
        val membersNamePairs =
          rules.map(r => r.mergeGroup.members.toSet -> r.name)
        given Order[Set[ImagingTimepoint]] = Order.by(_.toList)
        membersNamePairs.tail
          .foldLeft(
            NonEmptyMap
              .one[Set[ImagingTimepoint], TraceGroupId]
              .tupled(membersNamePairs.head)
          ) { case (acc, (members, name)) =>
            acc.apply(members) match {
              case None => acc.add(members -> name)
              case Some(establishedName) =>
                throw new Exception(
                  s"Group ($members) with name '$name' already maps to name '$establishedName'"
                )
            }
          }
      val rawLookup: Set[ImagingTimepoint] => Either[
        AtLeast2[List, (Set[ImagingTimepoint], TraceGroupId)],
        Option[TraceGroupId]
      ] =
        lookupBySubset(traceGroupIdByRegTimeSet.toSortedMap.toList)
      // Reduce the input to a "normal" Set, push it through the lookup, and then lift a good result.
      _.toSortedSet.pipe(rawLookup.map(_.map(TraceGroupMaybe.apply)))

    val initTraceId = getInitialTraceId(records.map(_.index))
    val traceIdsOffLimits =
      // Don't use any ROI index/ID as a trace ID.
      records.map(_.index.get).map(TraceId.unsafe).toNes
    computeNeighborsGraph(rules, pixels)(records)
      .strongComponentTraverser()
      .map(
        _.nodes
          .map(_.outer) // Get ROI IDs.
          .toList
          .toNel // each component as a nonempty list
          .getOrElse { throw new Exception("Empty component!") }
      ) // protected against by definition of component
      .toList
      .toNel // We want a nonempty list of components to accumulate errors
      .getOrElse {
        throw new Exception("No components!")
      } //  protected against by initial .toNel call on input ROIs
      // Recover the actual records for each component (consisting of IDs), using the lookup table/function.
      .traverse(_.traverse { i => lookupRecord.apply(i).toValidNel(i) })
      .fold(
        badIds =>
          // guarded against by construction of the lookup from records input
          throw new Exception(
            s"${badIds.length} ROI IDs couldn't be looked up! Here's one: ${badIds.head.show_}"
          ),
        _.toList
          .foldRight(initTraceId -> List.empty[InputRecordFate]) {
            case (recGroup, (currId, acc)) =>
              checkTraceId(traceIdsOffLimits)(currId)
              val newRecs: List[InputRecordFate] =
                processOneGroup(
                  discardIfNotInGroupOfInterest,
                  lookupRule,
                  lookupTraceGroupId
                )(recGroup, currId)
              val newTid =
                // Increment the trace ID if and only if any record in the current group is being emitted as output.
                if newRecs.forall(_.isLeft) then currId
                else TraceId.unsafe(NonnegativeInt(1) + currId.get)
              (newTid, newRecs ::: acc)
          }
          ._2
      )

  def processOneGroup(
      discardIfNotInGroupOfInterest: Boolean,
      lookupRule: TimepointExpectationLookup,
      lookupTraceGroupId: NonEmptySet[ImagingTimepoint] => Either[
        AtLeast2[List, (Set[ImagingTimepoint], TraceGroupId)],
        TraceGroupMaybe
      ]
  ): (NonEmptyList[InputRecord], TraceId) => List[InputRecordFate] =
    import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2.syntax.{
      map,
      remove,
      size,
      toNes,
      toSet
    }
    given Eq[RoiPartnersRequirementType] = Eq.fromUniversalEquals

    (recGroup, currId) =>
      AtLeast2
        .either(recGroup.map(_.index).toList.toSet)
        .fold(
          // singleton case
          Function.const {
            val singleInputRecord = recGroup.head
            val maybeOutputRecord = lookupRule
              // See if this record's timepoint is in one of the merge rules.
              .apply(recGroup.head.timepoint)
              // Generate an output record if either...
              //     a. No rule is found, and we're not discarding records from ungrouped timepoints, OR
              //     b. We're allowed to use records from timepoints to merge even if there's no merge to be done.
              .fold(!discardIfNotInGroupOfInterest)(
                _.requirement === RoiPartnersRequirementType.Lackadaisical
              )
              .option {
                val query = NonEmptySet.one(singleInputRecord.timepoint)
                val roiId = singleInputRecord.index
                lookupTraceGroupId(query) match {
                  case Left(multiHit) =>
                    // problem case --> siphon off separately
                    TraceGroupNameAmbiguity(
                      roiId,
                      query,
                      multiHit.map(_._2)
                    ).asLeft
                  case Right(traceGroupMaybe) =>
                    traceGroupMaybe.toOption match {
                      case None =>
                        val assignment =
                          TraceIdAssignment.UngroupedRecord(roiId, currId)
                        OutputRecord(singleInputRecord, assignment).asRight
                      case Some(groupId) =>
                        val assignment = TraceIdAssignment
                          .GroupedAndUnmerged(roiId, currId, groupId)
                        OutputRecord(singleInputRecord, assignment).asRight
                    }
                }
              }
            List(maybeOutputRecord).flatten
          },
          // at least two ROIs in group/component
          multiIds =>
            val maybeRepeatedTimepoints = recGroup
              .groupBy(_.timepoint)
              .view
              .toList
              .flatMap { (t, rs) =>
                AtLeast2
                  .either(rs.map(_.index).toList.toSet)
                  .toOption
                  .map(t -> _)
              }
              .toNel
            maybeRepeatedTimepoints match {
              case Some(reps) =>
                List(TimepointCollisionWithinTrace(multiIds, reps.toNem).asLeft)
              case None =>
                val observedTimes = recGroup.map(_.timepoint).toNes

                val (groupHasAllTimepoints, reqType) = recGroup.toList
                  .flatMap { r => lookupRule.apply(r.timepoint) }
                  .toNel
                  .map { rules =>
                    val nUniqueRules = rules.toList.toSet.size
                    if nUniqueRules =!= 1 then {
                      throw new Exception(
                        // This would be what would happen if one or more elements (timepoints) "bridge"
                        // the groups, i.e. one or more timepoints is used in more than one merge group,
                        // and a spot from such a timepoint acts as a bridge between what should be
                        // separate, independent connected components.
                        s"$nUniqueRules unique merge rules (not just 1) for single group! Timepoints: $observedTimes"
                      )
                    } else {
                      // Check here that either all timepoints for a group are present,
                      // or that the requirement type to keep a record is NOT conjunctive.
                      val rule = rules.head
                      val expectedTimes = rule.mergeGroup.members.toNes
                      val extraTimes = observedTimes -- expectedTimes
                      if extraTimes.nonEmpty then {
                        // This should be protected against by the construction of the neighbors graph;
                        // namely, we only draw an edge between points for which the pair of timepoints
                        // are part of the same group of timepoints to merge.
                        throw new Exception(
                          s"Extra time(s) not in merge rule $rule -- $extraTimes"
                        )
                      }
                      (
                        (expectedTimes -- observedTimes).isEmpty,
                        rule.requirement
                      )
                    }
                  }
                  .getOrElse {
                    // This is protected against by the construction of the neighbors graph; specifically, since we're
                    // working here with a nontrivial (multi-member) connected component, each pair of m
                    throw new Exception(
                      s"No merge rules found for multi-member group! ROI IDs: ${multiIds}. Timepoints: $observedTimes"
                    )
                  }
                if groupHasAllTimepoints || reqType =!= RoiPartnersRequirementType.Conjunctive
                then
                  val emitElem
                      : InputRecord => InputRecordFate = lookupTraceGroupId(
                    observedTimes
                  ) match {
                    case Left(multiHit) =>
                      (r: InputRecord) =>
                        TraceGroupNameAmbiguity(
                          r.index,
                          observedTimes,
                          multiHit.map(_._2)
                        ).asLeft
                    case Right(groupIdOpt) =>
                      val groupId = groupIdOpt.toOption.getOrElse {
                        // TODO: should this be made a valid case?
                        throw new Exception(
                          s"No trace group ID found for multi-timepoint group (${})"
                        )
                      }
                      (r: InputRecord) =>
                        val partners = multiIds.remove(r.index)
                        val assignment = TraceIdAssignment.GroupedAndMerged(
                          r.index,
                          currId,
                          groupId,
                          partners,
                          groupHasAllTimepoints
                        )
                        OutputRecord(r, assignment).asRight
                  }
                  recGroup.map(emitElem).toList
                else
                  List() // The group lacked all the required regional imaging timepoints, and that's required if we're at this conditional.
            }
        )

  final case class TimepointCollisionWithinTrace(
      roiIds: AtLeast2[Set, RoiIndex],
      repeatedTimesWithinTrace: NonEmptyMap[
        ImagingTimepoint,
        AtLeast2[Set, RoiIndex]
      ]
  )

  final case class TraceGroupNameAmbiguity(
      roiId: RoiIndex,
      timesQuery: NonEmptySet[ImagingTimepoint],
      traceGroupIds: AtLeast2[List, TraceGroupId]
  )

  private type AssignmentNotPossible = TimepointCollisionWithinTrace |
    TraceGroupNameAmbiguity

  private type InputRecordFate = Either[
    TimepointCollisionWithinTrace | TraceGroupNameAmbiguity,
    OutputRecord
  ]

  /** Helpers for working with the case of a record being unable to be processed
    */
  object InputRecordFate:
    import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2.syntax.toList
    import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
    import at.ac.oeaw.imba.gerlich.gerlib.json.JsonValueWriter
    import at.ac.oeaw.imba.gerlich.gerlib.json.instances.all.given

    given (
        upickle.default.Writer[TimepointCollisionWithinTrace],
        upickle.default.Writer[TraceGroupNameAmbiguity]
    ) => upickle.default.Writer[AssignmentNotPossible] =
      upickle.default.writer[ujson.Obj].comap(toJson)

    private def toJson(impossibility: AssignmentNotPossible): ujson.Obj =
      impossibility match {
        case collision: TimepointCollisionWithinTrace => toJson(collision)
        case ambiguity: TraceGroupNameAmbiguity       => toJson(ambiguity)
      }

    private def toJson(ambiguity: TraceGroupNameAmbiguity): ujson.Obj =
      ujson.Obj(
        "roiId" -> ambiguity.roiId.get,
        "groupTimes" -> ambiguity.timesQuery.toNonEmptyList.toList
          .sorted(using Order[ImagingTimepoint].toOrdering)
          .map(_.asJson),
        "groupIds" -> ambiguity.traceGroupIds.toList.map(_.get)
      )

    private def toJson(collision: TimepointCollisionWithinTrace): ujson.Obj =
      ujson.Obj(
        "roiIds" -> collision.roiIds.toList.map(_.get),
        "repeats" -> collision.repeatedTimesWithinTrace.toSortedMap.unsorted
          .map { (t, rois) =>
            // NB: here we convert the timepoint key to text to comply with JSON.
            t.show_ -> rois.toList
              .map(_.get)
              .sorted(Order[Int :| Not[Negative]].toOrdering)
          }
      )

    given (w: upickle.default.Writer[AssignmentNotPossible])
      => upickle.default.Writer[TraceGroupNameAmbiguity] =
      w.narrow

    given (w: upickle.default.Writer[AssignmentNotPossible])
      => upickle.default.Writer[TimepointCollisionWithinTrace] =
      w.narrow

    given writerForImpossibleAssignments
        : upickle.default.Writer[List[AssignmentNotPossible]] =
      // NB: need more manual derivation than usual, since we're using a union type.
      // https://github.com/com-lihaoyi/upickle/issues/505
      // https://github.com/com-lihaoyi/upickle/issues/481
      upickle.default.writer[ujson.Arr].comap(_.map(toJson))
  end InputRecordFate

  def workflow(
      roundsConfig: ImagingRoundsConfiguration,
      roisFile: os.Path,
      pixels: Pixels3D,
      outputFile: os.Path,
      skipsFile: os.Path
  ): Unit = {
    import InputRecord.given
    import fs2.data.text.utf8.*
    import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2.syntax.toSet

    given CsvRowDecoder[ImagingChannel, String] =
      getCsvRowDecoderForImagingChannel(SpotChannelColumnName)

    val readRois: IO[List[InputRecord]] = for
      _ <- IO { logger.info(s"Reading ROIs file: $roisFile") }
      rois <- readCsvToCaseClasses[InputRecord](roisFile)
    yield rois

    val assignIds: List[InputRecord] => IO[List[InputRecordFate]] =
      _.toNel match {
        case None =>
          IO {
            logger.error(s"No input record parsed from ROIs file ($roisFile)!")
          }.as(List())
        case Some(records) =>
          IO.pure {
            roundsConfig.mergeRules match {
              case None =>
                // No merger of ROIs for tracing, so no need to find group ID or partners.
                val initTraceId = getInitialTraceId(records.map(_.index))
                val traceIdsOffLimits =
                  records.map(r => TraceId(r.index.get)).toNes
                NonnegativeInt
                  .indexed(records)
                  .map { (r, i) =>
                    val newTid = TraceId.unsafe(i + initTraceId.get)
                    checkTraceId(traceIdsOffLimits)(newTid)
                    OutputRecord(
                      r,
                      TraceIdAssignment.UngroupedRecord(r.index, newTid)
                    ).asRight
                  }
                  .toList
              case Some(rules) =>
                labelRecordsWithTraceId(
                  rules,
                  roundsConfig.discardRoisNotInGroupsOfInterest,
                  pixels
                )(records)
            }
          }
      }

    val writeOutputs: List[InputRecordFate] => IO[List[os.Path]] = _ match {
      case Nil => IO { logger.error("No output to write!") }.as(List())
      case inputFates =>
        import OutputRecord.given
        import InputRecordFate.given

        given CellEncoder[FieldOfViewLike]:
          override def apply(cell: FieldOfViewLike): String = cell match {
            case fov: FieldOfView => CellEncoder[FieldOfView].apply(fov)
            case pos: PositionName =>
              OneBasedFourDigitPositionName
                .fromPositionName(pos)
                .fold(
                  msg => throw new RuntimeException(msg),
                  identity
                )
          }
        given CsvRowEncoder[ImagingChannel, String] =
          // for derivation of CsvRowEncoder[ImagingContext, String]
          SpotChannelColumnName.toNamedEncoder

        val (skips, records) = Alternative[List].separate(inputFates)

        val writeMainOutput: IO[os.Path] = for
          _ <- IO { logger.info(s"Writing main output file: $outputFile") }
          _ <- fs2.Stream
            .emits(
              records
                .sortBy(_.inputRecord.index)(using Order[RoiIndex].toOrdering)
                .toList
            )
            .through(writeCaseClassesToCsv[OutputRecord](outputFile))
            .compile
            .drain
        yield outputFile

        val writeSkipsOutput: IO[os.Path] =
          import InputRecordFate.given
          for
            _ <- IO { logger.info(s"Writing skips file: $skipsFile") }
            _ <- IO {
              os.write(
                skipsFile,
                upickle.default.write(skips),
                createFolders = true
              )
            }
          yield skipsFile

        List(writeMainOutput, writeSkipsOutput).sequence
    }

    val prog = for
      rois <- readRois
      fates <- assignIds(rois)
      outpaths <- writeOutputs(fates)
      _ <- IO {
        logger.info(
          s"Wrote ${outpaths.length} path(s): ${outpaths mkString ", "}"
        )
      }
    yield ()

    prog.unsafeRunSync()
    logger.info("Done!")
  }

  final case class OutputRecord(
      inputRecord: InputRecord, // NB: this part of the record contains the ACTUAL merge partners (if any).
      assignment: TraceIdAssignment
  ):
    def index: RoiIndex = inputRecord.index
    def context: ImagingContext = inputRecord.context
    def centroid: Centroid[Double] = inputRecord.centroid
    def box: BoundingBox = inputRecord.box

  object OutputRecord:
    given (
        encRoiId: CellEncoder[RoiIndex],
        encContext: CsvRowEncoder[ImagingContext, String],
        encCentroid: CsvRowEncoder[Centroid[Double], String],
        encBox: CsvRowEncoder[BoundingBox, String],
        encNuc: CellEncoder[NuclearDesignation],
        encTid: CellEncoder[TraceId],
        encTraceGroupId: CellEncoder[TraceGroupMaybe],
        encPartnersFlag: CellEncoder[Boolean]
    ) => CsvRowEncoder[OutputRecord, String] = new:
      override def apply(elem: OutputRecord): RowF[Some, String] =
        val idRow = RoiIndexColumnName.write(elem.index)
        val ctxRow = encContext(elem.context)
        val centerRow = encCentroid(elem.centroid)
        val boxRow = encBox(elem.box)
        val mergeInputsRow =
          MergeContributorsColumnNameForAssessedRecord.write(
            elem._1.maybeMergeInputs
          )
        val nucRow =
          elem._1.maybeNucleusDesignation match {
            case None =>
              RowF(
                values = NonEmptyList.one(""),
                headers =
                  Some(NonEmptyList.one(NucleusDesignationColumnName.value))
              )
            case Some(nuclearDesignation) =>
              NucleusDesignationColumnName.write(nuclearDesignation)
          }
        val encAssignment: CsvRowEncoder[TraceIdAssignment, String] =
          getCsvRowEncoderForTraceIdAssignmentWithoutRoiIndex
        val traceIdAssignmentRow = encAssignment(elem.assignment)
        idRow |+| ctxRow |+| centerRow |+| boxRow |+| mergeInputsRow |+| nucRow |+| traceIdAssignmentRow
  end OutputRecord

  final case class InputRecord(
      index: RoiIndex,
      context: ImagingContext,
      centroid: Centroid[Double],
      box: BoundingBox,
      maybeMergeInputs: Set[
        RoiIndex
      ], // may be empty, as the input collection is possibly a mix of singletons and merge results
      maybeNucleusDesignation: Option[
        NuclearDesignation
      ] // Allow the program to operate on non-nuclei-filtered ROIs.
  ):
    final def timepoint: ImagingTimepoint = context.timepoint

  /** Helpers for working with this program's input records */
  object InputRecord:
    given (
        decIndex: CellDecoder[RoiIndex],
        decContext: CsvRowDecoder[ImagingContext, String],
        decCentroid: CsvRowDecoder[Centroid[Double], String],
        decBox: CsvRowDecoder[BoundingBox, String],
        decNuclus: CellDecoder[NuclearDesignation]
    ) => CsvRowDecoder[InputRecord, String] = new:
      override def apply(row: RowF[Some, String]): DecoderResult[InputRecord] =
        val spotNel = summon[CsvRowDecoder[IndexedDetectedSpot, String]](row)
          .leftMap(e => s"Cannot decode spot from row ($row): ${e.getMessage}")
          .toValidatedNel
        val mergeInputsNel =
          MergeContributorsColumnNameForAssessedRecord.from(row)
        val nucNel =
          val key = NucleusDesignationColumnName.value
          row.apply(key) match {
            // Allow the program to operate on non-nuclei-filtered ROIs.
            case None | Some("") => Option.empty.validNel
            case Some(s) =>
              decNuclus(s)
                .bimap(
                  e => s"Cannot decode spot from row ($row): ${e.getMessage}",
                  _.some
                )
                .toValidatedNel
          }
        (spotNel, mergeInputsNel, nucNel)
          .mapN { (spot, maybeMergeIndices, maybeNucleus) =>
            InputRecord(
              spot.index,
              spot.context,
              spot.centroid,
              spot.box,
              maybeMergeIndices,
              maybeNucleus
            )
          }
          .toEither
          .leftMap { messages =>
            DecoderError(
              s"${messages.length} error(s) reading row ($row):\n${messages.mkString_("\n")}"
            )
          }
  end InputRecord
end AssignTraceIds
