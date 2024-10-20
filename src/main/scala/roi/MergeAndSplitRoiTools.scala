package at.ac.oeaw.imba.gerlich.looptrace
package roi

import cats.*
import cats.data.{ EitherNel, NonEmptyList, NonEmptyMap, NonEmptySet, ValidatedNel }
import cats.syntax.all.*
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.geometry.{ Centroid, DistanceThreshold, PiecewiseDistance, ProximityComparable }
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.geometry.syntax.*
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
object MergeAndSplitRoiTools:
    private[looptrace] type PostMergeRoi = IndexedDetectedSpot | MergedRoiRecord
    import PostMergeRoi.*

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

    def mergeRois(buildNewBox: (Point3D, NonEmptyList[BoundingBox]) => BoundingBox)(rois: List[MergerAssessedRoi]): MergeResult = 
        rois match {
            case Nil => (List(), List(), List(), List())
            case _ => 
                def incrementIndex: RoiIndex => RoiIndex = i => RoiIndex.unsafe(i.get + 1)
                val pool = rois.map{ r => r.index -> r }.toMap
                given Ordering[RoiIndex] = summon[Order[RoiIndex]].toOrdering
                val initNewIndex = incrementIndex(rois.map(_.index).max)
                val ((allErrored, allSkipped, allMerged), _) = 
                    rois.foldRight(((List.empty[MergeError], List.empty[IndexedDetectedSpot], List.empty[MergedRoiRecord]), initNewIndex)){
                        case (r, ((accErr, accSkip, accMerge), currentMergeIndex)) => 
                            considerOneMerge(pool, buildNewBox)(currentMergeIndex, r) match {
                                case None => 
                                    // no merge action; simply eliminate the empty mergePartners collection
                                    val idxSpot = IndexedDetectedSpot(r.index, r.context, r.centroid, r.box)
                                    (accErr, idxSpot :: accSkip, accMerge) -> currentMergeIndex
                                case Some(Left(errors)) => 
                                    // error case
                                    ((r, errors) :: accErr, accSkip, accMerge) -> currentMergeIndex
                                case Some(Right(rec)) =>
                                    // merge action
                                    (accErr, accSkip, rec :: accMerge) -> incrementIndex(currentMergeIndex)
                        }
                    }
                val allContrib: List[MergeContributorRoi] = allMerged
                    .flatMap{ roi => roi.contributors.toList.map(_ -> roi.index) }
                    .map{ (contribIndex, mergedIndex) => 
                        val original = pool.getOrElse(
                            contribIndex, 
                            throw new Exception(s"Cannot find original ROI for alleged contributor index ${contribIndex.show_}")
                        )
                        MergeContributorRoi(
                            contribIndex, 
                            original.context, 
                            original.centroid, 
                            original.box,
                            mergedIndex, 
                        )
                    }
                    .sortBy(_.index)
                (allErrored, allSkipped, allContrib, allMerged)                
        }
    
    /** Do the merge for a single ROI record. */
    private[looptrace] def considerOneMerge(
        pool: Map[RoiIndex, MergerAssessedRoi], 
        buildNewBox: (Point3D, NonEmptyList[BoundingBox]) => BoundingBox,
    )(potentialNewIndex: RoiIndex, roi: MergerAssessedRoi): Option[Either[NonEmptyList[String], MergedRoiRecord]] = 
        roi.mergeNeighbors.toList.toNel.map(
            _.traverse{ i => 
                pool.get(i)
                    .toRight(s"Missing ROI index: ${i.get}")
                    .toValidatedNel
            }
            .toEither
            .flatMap{ partners => 
                val contexts = partners.map(_.context).toNes
                (contexts.size === 1)
                    .validatedNel(
                        s"${contexts.size} unique imaging context (not just 1) in ROI group to merge", 
                        partners.head.context
                    )
                    .map{ ctx => 
                        val newCenter: Point3D = partners.map(_.centroid.asPoint).centroid
                        val newBox: BoundingBox = buildNewBox(newCenter, partners.map(_.box))
                        MergedRoiRecord(
                            potentialNewIndex, 
                            ctx, 
                            Centroid.fromPoint(newCenter), 
                            newBox, 
                            partners.map(_.index).toNes,
                        )
                    }
                    .toEither
            }
        )

    private type MergeError = (MergerAssessedRoi, ErrorMessages)
    
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

    private type MergeResult = (
        List[MergeError], // errors
        List[IndexedDetectedSpot], // non-participants in merge
        List[MergeContributorRoi], // contributors to merge
        List[MergedRoiRecord], // merge outputs
    )
end MergeAndSplitRoiTools