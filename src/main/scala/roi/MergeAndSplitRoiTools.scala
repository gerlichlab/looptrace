package at.ac.oeaw.imba.gerlich.looptrace
package roi

import scala.collection.immutable.SortedMap
import cats.*
import cats.data.{ EitherNel, NonEmptyList, NonEmptyMap, NonEmptySet, ValidatedNel }
import cats.syntax.all.*
import io.github.iltotore.iron.constraint.collection.given
import mouse.boolean.*
import com.typesafe.scalalogging.LazyLogging

import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2
import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2.syntax.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.{ Centroid, DistanceThreshold, PiecewiseDistance, ProximityComparable }
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.geometry.syntax.*
import at.ac.oeaw.imba.gerlich.gerlib.graph.{ SimplestGraph, buildSimpleGraph }
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.space.{ BoundingBox, Point3D }
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.NontrivialProximityFilter
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.UniversalProximityPermission
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.UniversalProximityProhibition
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.SelectiveProximityPermission
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.SelectiveProximityProhibition
import at.ac.oeaw.imba.gerlich.gerlib.imaging.FieldOfViewLike
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.ProximityFilterStrategy
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint

/** Tools for merging ROIs */
object MergeAndSplitRoiTools extends LazyLogging:
    private[looptrace] type PostMergeRoi = IndexedDetectedSpot | MergedRoiRecord
    import PostMergeRoi.*

    type RoiMergeBag = AtLeast2[Set, RoiIndex]

    /** Facilitate access to the components of a ROI's imaging context. */
    object PostMergeRoi:
        extension (roi: PostMergeRoi)
            def context(using 
                AdmitsImagingContext[IndexedDetectedSpot],
                AdmitsImagingContext[MergedRoiRecord],
            ): ImagingContext = 
                import AdmitsImagingContext.*
                roi match {
                    case unmerged: IndexedDetectedSpot => unmerged.imagingContext
                    case merged: MergedRoiRecord => merged.imagingContext
                }

        given AdmitsRoiIndex[PostMergeRoi] with
            override def getRoiIndex = (_: PostMergeRoi) match {
                case roi: IndexedDetectedSpot => roi.index
                case roi: MergedRoiRecord => roi.index
            }

        /** Regardless of sepecific ROI subtype, get the center point and the bounding box. */
        def getCenterAndBox = (_: PostMergeRoi) match {
            case roi: MergedRoiRecord => roi.centroid -> roi.box
            case roi: IndexedDetectedSpot => roi.centroid -> roi.box
        }
    end PostMergeRoi

    def assessForMerge(minDist: DistanceThreshold)(rois: List[IndexedDetectedSpot]): EitherNel[String, List[MergerAssessedRoi]] = 
        import ProximityComparable.proximal
        given proxComp: ProximityComparable[IndexedDetectedSpot] = 
            DistanceThreshold.defineProximityPointwise(minDist)(_.centroid.asPoint)
        val lookup: Map[RoiIndex, Set[RoiIndex]] = 
            rois.groupBy(_.context) // Only merge ROIs from the same context (FOV, time, channel).
                .values
                .flatMap(_.combinations(2).flatMap{
                    case r1 :: r2 :: Nil => 
                        (r1 `proximal` r2).option{ Map(r1.index -> Set(r2.index), r2.index -> Set(r1.index)) }
                    case notPair => throw new Exception(s"Got ${notPair.length} element(s) when taking pairs!")
                })
                .toList
                .combineAll
        val (errors, records) = Alternative[List].separate(
            rois.map{ r => MergerAssessedRoi.build(r, lookup.getOrElse(r.index, Set.empty)) }
        )
        errors.toNel.toLeft(records)
    
    private def buildMutualExclusionLookup[A, G, K: Order](
        items: List[A], 
        getGroupKey: A => G,
        useEligiblePair: (A, A) => EitherNel[String, Boolean], 
        getPoint: A => Point3D, 
        minDist: DistanceThreshold,
        getItemKey: A => K,
    ): (List[((K, K), NonEmptyList[String])], Map[K, NonEmptySet[K]]) = 
        import ProximityComparable.proximal
        given proxComp: ProximityComparable[A] = DistanceThreshold.defineProximityPointwise(minDist)(getPoint)
        val (errors, indexes): (List[((K, K), NonEmptyList[String])], Map[K, NonEmptySet[K]]) = 
            items.groupBy(getGroupKey)
                .values
                .foldRight(List.empty[((K, K), NonEmptyList[String])] -> Map.empty[K, NonEmptySet[K]]){ case (group, (accBad, accGood)) => 
                    val (newBads, closePairs) = Alternative[List].separate(
                        group.combinations(2)
                            .flatMap{
                                case a1 :: a2 :: Nil => 
                                    useEligiblePair(a1, a2).fold(
                                        es => ((getItemKey(a1) -> getItemKey(a2)) -> es).asLeft.some, 
                                        continue => (continue && (a1 `proximal` a2)).option{ (getItemKey(a1) -> getItemKey(a2)).asRight }
                                    )
                                case nonPair => throw new Exception(s"Got ${nonPair.length} items when taking pairs!")
                            }
                            .toList
                    )
                    val newGoods = closePairs.foldLeft(Map.empty[K, NonEmptySet[K]]){ case (acc, (k1, k2)) => 
                        val v1 = acc.get(k1).fold(NonEmptySet.one[K])(_.add)(k2)
                        val v2 = acc.get(k2).fold(NonEmptySet.one[K])(_.add)(k1)
                        acc ++ Map(k1 -> v1, k2 -> v2)
                    }
                    (newBads |+| accBad, accGood |+| newGoods)
                }
        errors -> indexes

    /** Use the given grouping strategy and minimal separation threshold to determine which ROIs mutually exclude each other due to lack of identifiability. */
    def assessForMutualExclusion(proximityFilterStrategy: NontrivialProximityFilter)(rois: List[PostMergeRoi])(using Eq[FieldOfViewLike], AdmitsRoiIndex[PostMergeRoi]): 
        (List[UnidentifiableRoi], List[PostMergeRoi]) = 
        import IndexedDetectedSpot.given
        import AdmitsRoiIndex.*
        
        // Get the unique identifier for a ROI.
        val getItemKey: PostMergeRoi => RoiIndex = _.roiIndex
        
        // The minimum separation required for two ROIs to be considered distinct/identifiable.
        val minDist: DistanceThreshold = PiecewiseDistance.ConjunctiveThreshold(proximityFilterStrategy.minSpotSeparation)

        // Decide whether to do the distance computation for the two ROIs, based on their timepoints and the proximity grouping strategy.
        val usePairBasedOnGroup: (PostMergeRoi, PostMergeRoi) => EitherNel[String, Boolean] = 
            proximityFilterStrategy match {
                // For universal prohibition, check all grouped pairs.
                case UniversalProximityProhibition(_) => (_1, _2) => true.asRight
                // For selective strategies, map imaging timepoint to group ID, then compare group IDs for each ROI pair.
                case strat: (SelectiveProximityPermission | SelectiveProximityProhibition) => 
                    val minSep = PiecewiseDistance.ConjunctiveThreshold(strat.minSpotSeparation)
                    val groupIds: NonEmptyMap[ImagingTimepoint, Int] = 
                        strat.grouping.zipWithIndex.flatMap{ (g, i) => g.toNonEmptyList.map(_ -> i) }.toNem
                    val getGroupId = (roi: PostMergeRoi) => 
                        val t = roi.context.timepoint
                        groupIds(t).toRight(s"No group ID for ROI's timepoint (${t.show_})")
                    val checkGroupIds: (Int, Int) => Boolean = strat match {
                        case _: SelectiveProximityPermission => 
                            // For selective permission, check pairs from different groups, since grouped timepoints are allowed to be close.
                            (i1, i2) => i1 =!= i2
                        case _: SelectiveProximityProhibition => 
                            // For selective prohibition, check pairs from the same group, since grouped timepoints may not be close.
                            (i1, i2) => i1 === i2
                    }
                    (a, b) => 
                        val aIdNel = getGroupId(a).toValidatedNel
                        val bIdNel = getGroupId(b).toValidatedNel
                        (aIdNel, bIdNel).mapN(checkGroupIds).toEither
            }

        // To be able to mutually exclude one another, ROIs must be from the same field of view and imaging channel.
        val getGroupKey = (roi: PostMergeRoi) => roi.context.fieldOfView -> roi.context.channel
        //  The overall decision to use a pair of ROIs is a conjunction of the decision based on grouping equivalence and the one based on timepoint nonequivalence.
        val usePair: (PostMergeRoi, PostMergeRoi) => EitherNel[String, Boolean] = (a, b) => 
            if a.context.timepoint === b.context.timepoint
            // To be able to mutually exclude one another, ROIs must be from different timepoints (same timepoint ROIs should perhaps merge).
            then false.asRight
            else  usePairBasedOnGroup(a, b)
        // Create the lookup table, mapping an individual record's ROI index to the collection of its too-close neighbors.
        val (lookupBuildErrors, lookup) = buildMutualExclusionLookup(rois, getGroupKey, usePair, getCenterAndBox.map(_._1.asPoint), minDist, getItemKey)
        lookupBuildErrors.toNel match {
            case None => Alternative[List].separate(rois.map{ roi => 
                val key = getItemKey(roi)
                lookup.get(key).toLeft(roi).leftMap{ tooClose => 
                    val (pt, box) = getCenterAndBox(roi)
                    UnidentifiableRoi(key, roi.context, pt, box, tooClose)
                }
            })
            case Some(errors) => throw new Exception(s"${errors.length} ROI pairs for which proximity could not be assessed. Here's an example: ${errors.head}")
        }

    /** Eliminate the trivial filtration strategy as a possibility, and defer to the more general implementation. */
    def assessForMutualExclusion(proximityFilterStrategy: ProximityFilterStrategy)(rois: List[PostMergeRoi])(using Eq[FieldOfViewLike], AdmitsRoiIndex[PostMergeRoi]): 
        (List[UnidentifiableRoi], List[PostMergeRoi]) = 
        proximityFilterStrategy match {
            case UniversalProximityPermission => List() -> rois
            case strat: NontrivialProximityFilter => assessForMutualExclusion(strat)(rois)
        }

    /** Wrap the simple graph builder to account for the (bad) possibility of duplicate records by key. */
    def buildGraph[Key: Order, Record](
        getKey: Record => Key, 
        getNeighbors: Record => Set[Key],
    ): List[Record] => Either[NonEmptyMap[Key, List[(Record, Int)]], SimplestGraph[Key]] = records => 
        given Ordering[Key] = summon[Order[Key]].toOrdering
        val (bads, goods) = records.zipWithIndex.foldRight(SortedMap.empty[Key, List[(Record, Int)]] -> Map.empty[Key, Set[Key]]){
            case (pair@(rec, i), (bads, goods)) => 
                val k = getKey(rec)
                if goods contains k 
                then (bads + (k -> (pair :: bads.getOrElse(k, List.empty))), goods)
                else (bads, goods + (k -> getNeighbors(rec)))
        }
        NonEmptyMap.fromMap(bads)
            .toLeft(goods)
            .map{ adj => buildSimpleGraph(adj.toList) }

    /**
     * Generate the merge contributors, merge results, and singleton ROIs from the collection of merge-determined ROIs.
     * 
     * @param buildNewBox How to construct a new bounding box for a merge result, given the coordinate for the new center and 
     *     the collection of bounding boxes of the ROIs which contributed to the merge result; this collection is useful, for 
     *     example, to determine the minimum or maximum box size, or perhaps something like the greatest extent encompassed 
     *     by the union of the boxes around the ROIs contributing to the merge
     * @param rois The collection of ROIs for which merge partners(s) (or not) have been determined
     * @return A list of singleton ROIs, a list of merge inputs, and a list of merge outputs
     */
    def mergeRois(buildNewBox: (Point3D, NonEmptyList[BoundingBox]) => BoundingBox)(rois: List[MergerAssessedRoi]): (
        List[IndexedDetectedSpot], // non-participants in merge
        List[MergeContributorRoi], // contributors to merge
        List[MergedRoiRecord], // merge outputs
    ) = 
        if rois.isEmpty then (List(), List(), List()) 
        else buildGraph((_: MergerAssessedRoi).index, (_: MergerAssessedRoi).mergeNeighbors)(rois) match {
            case Left(repeats) => 
                throw new Exception(
                    s"${repeats.size} case(s) of repeated key in ROI records. Here they are mapped, to collection of pairs of record and record number (0-based): $repeats"
                )
            case Right(graph) => 
                def incrementIndex: RoiIndex => RoiIndex = i => RoiIndex.unsafe(i.get + 1)
                val pool = rois.map{ r => r.index -> r }.toMap
                given Ordering[RoiIndex] = summon[Order[RoiIndex]].toOrdering
                val initNewIndex = incrementIndex(rois.map(_.index).max)
                logger.debug(s"Initial / first eligible index to use for merge result ROIs: ${initNewIndex.show_}")
                val (allSkipped, allMerged, _) = 
                    graph.strongComponentTraverser().foldLeft((List.empty[IndexedDetectedSpot], List.empty[MergedRoiRecord], initNewIndex)){
                        case ((accSingle, accMerged, currentMergeIndex), component) => 
                            component.nodes.map(_.outer).toList match {
                                case Nil => 
                                    // error case
                                    throw new Exception("Empty graph component!")
                                case id :: Nil => 
                                    pool.get(id)
                                        .map{ r => 
                                            val idxSpot = IndexedDetectedSpot(r.index, r.context, r.centroid, r.box)
                                            (idxSpot :: accSingle, accMerged, currentMergeIndex)
                                        }
                                        .getOrElse{ throw new Exception(s"Failed to look up ROI for singleton ID: ${id.show_}") }
                                case id1 :: id2 :: rest => 
                                    // merge case
                                    NonEmptyList(id1, id2 :: rest)
                                        .traverse{ i => 
                                            // should never happen, since we create the lookup pool within this function
                                            pool.get(i).toRight(s"Missing ROI index: ${i.show_}").toEitherNel 
                                        }
                                        .flatMap{ groupRois => 
                                            // Define the new center and box, and check that each ROI comes from the same imaging context.
                                            val newCenter: Point3D = groupRois.map(_.centroid.asPoint).centroid
                                            val newBox: BoundingBox = buildNewBox(newCenter, groupRois.map(_.box))
                                            val contexts = groupRois.map(_.context).toList.toSet
                                            val errorOrRecord = for {
                                                ctx <- (contexts.size === 1).either(
                                                    s"${contexts.size} unique imaging context (not just 1) in ROI group to merge",
                                                    groupRois.head.context
                                                )
                                                groupIds <- AtLeast2.either(groupRois.toList.map(_.index).toSet)
                                                _ <- 
                                                    if pool contains currentMergeIndex 
                                                    then s"Cannot use ${currentMergeIndex.show_} as merge record index; it's already used".asLeft
                                                    else ().asRight
                                            } yield MergedRoiRecord(
                                                currentMergeIndex, 
                                                ctx, 
                                                Centroid.fromPoint(newCenter), 
                                                newBox, 
                                                groupIds,
                                            )
                                            errorOrRecord.leftMap(NonEmptyList.one)
                                        }
                                        .fold(
                                            errors => throw new Exception(s"${errors.size} error(s) merging ROIs: ${errors}"),
                                            mergedRecord => 
                                                val mergeText = mergedRecord.contributors.toList.sorted.map(_.show_).mkString(";")
                                                logger.debug(s"Merged $mergeText --> ${mergedRecord.index.show_}")
                                                ((accSingle, mergedRecord :: accMerged, incrementIndex(mergedRecord.index)))
                                        )
                            }
                        }
                val allContrib: List[MergeContributorRoi] = getMergeContributorRois(pool, allMerged)
                (allSkipped.sortBy(_.index), allContrib.sortBy(_.index), allMerged.sortBy(_.index))
        }

    // Using the merge results, recover the contributing records.
    private[MergeAndSplitRoiTools] def getMergeContributorRois(
        mergeInputPool: Map[RoiIndex, MergerAssessedRoi], 
        merged: List[MergedRoiRecord],
    ): List[MergeContributorRoi] = 
        merged.foldRight(Map.empty[RoiIndex, RoiIndex]){ 
            (roi, mergeIndexByContribIndex) => 
                roi.contributors.toSet.foldRight(mergeIndexByContribIndex){ 
                    (i, acc) => acc.get(i) match {
                        case None => acc + (i -> roi.index)
                        case Some(prev) => throw new Exception(
                            s"Index ${i.show_} already contributed to merger ${prev.show_}"
                        )
                    }
                }
        }
        .toList
        .map{ (contribIndex, mergeOutput) => 
            val original = mergeInputPool.getOrElse(
                contribIndex, 
                throw new Exception(
                    s"Cannot find original ROI for alleged contributor index ${contribIndex.show_}"
                )
            )
            MergeContributorRoi(
                contribIndex, 
                original.context, 
                original.centroid, 
                original.box,
                mergeOutput, 
            )
        }
    
    final case class IndexedDetectedSpot(
        index: RoiIndex, 
        context: ImagingContext, 
        centroid: Centroid[Double],
        box: BoundingBox, 
    )

    private[looptrace] object IndexedDetectedSpot:
        given AdmitsRoiIndex[IndexedDetectedSpot] = AdmitsRoiIndex.instance(_.index)
        given AdmitsImagingContext[IndexedDetectedSpot] = AdmitsImagingContext.instance(_.context)
    end IndexedDetectedSpot

end MergeAndSplitRoiTools