package at.ac.oeaw.imba.gerlich.looptrace
package roi

import cats.*
import cats.data.{ NonEmptyList, NonEmptySet, ValidatedNel }
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

/** Tools for merging ROIs */
object MergeAndSplitRoiTools:
    private type PostMergeRoi = IndexedDetectedSpot | MergedRoiRecord

    def assessForMerge(rois: List[DetectedSpotRoi]): List[MergerAssessedRoi] = ???

    private def buildMutualExclusionLookup[A, G, K: Order](
        items: List[A], 
        getGroupKey: A => G,
        useEligiblePair: (A, A) => Boolean, 
        getPoint: A => Point3D, 
        minDist: DistanceThreshold,
        getItemKey: A => K,
    ): Map[K, NonEmptySet[K]] = 
        import ProximityComparable.proximal
        given proxComp: ProximityComparable[A] = DistanceThreshold.defineProximityPointwise(minDist)(getPoint)
        items.groupBy(getGroupKey)
            .values
            .map{ group => 
                val closePairs: List[(K, K)] = group.combinations(2)
                    .flatMap{
                        case a1 :: a2 :: Nil => 
                            (useEligiblePair(a1, a2) && (a1 `proximal` a2))
                                .option{ getItemKey(a1) -> getItemKey(a2) }
                        case nonPair => throw new Exception(s"Got ${nonPair.length} items when taking pairs!")
                    }
                    .toList
                closePairs.foldLeft(Map.empty[K, NonEmptySet[K]]){ case (acc, (k1, k2)) => 
                    val v1 = acc.get(k1).fold(NonEmptySet.one[K])(_.add)(k2)
                    val v2 = acc.get(k2).fold(NonEmptySet.one[K])(_.add)(k1)
                    acc ++ Map(k1 -> v1, k2 -> v2)
                }
            }
            .toSeq
            .combineAll

    /** Use the given grouping strategy and minimal separation threshold to determine which ROIs mutually exclude each other due to lack of identifiability. */
    def assessForMutualExclusion(proximityFilterStrategy: NontrivialProximityFilter)(rois: List[PostMergeRoi])(using Eq[FieldOfViewLike], AdmitsRoiIndex[PostMergeRoi]): 
        (List[UnidentifiableRoi], List[PostMergeRoi]) = 
        import IndexedDetectedSpot.given
        import AdmitsRoiIndex.*
        
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
        
        val usePair: (PostMergeRoi, PostMergeRoi) => Boolean = (a, b) => 
            a.context.fieldOfView === b.context.fieldOfView && a.context.timepoint =!= b.context.timepoint
        val getItemKey: PostMergeRoi => RoiIndex = _.roiIndex
        val getCenterAndBox = (_: PostMergeRoi) match {
            case roi: MergedRoiRecord => roi.centroid -> roi.box
            case (_, roi: DetectedSpotRoi) => roi.centroid -> roi.box
        }
        val minDist: DistanceThreshold = PiecewiseDistance.ConjunctiveThreshold(proximityFilterStrategy.minSpotSeparation)

        val getGroupKey: PostMergeRoi => Option[Int] = proximityFilterStrategy match {
            case UniversalProximityProhibition(minSpotSeparation) => ???
            case SelectiveProximityPermission(minSpotSeparation, grouping) => ???
            case SelectiveProximityProhibition(minSpotSeparation, grouping) => ???
        }
        
        val lookup = buildMutualExclusionLookup(rois, getGroupKey, usePair, getCenterAndBox.map(_._1.asPoint), minDist, getItemKey)

        Alternative[List].separate(rois.map{ roi => 
            val key = getItemKey(roi)
            lookup.get(key).toLeft(roi).leftMap{ tooClose => 
                val (pt, box) = getCenterAndBox(roi)
                UnidentifiableRoi(key, roi.context, pt, box, tooClose)
            }
        })

    /** Eliminate the trivial filtration strategy as a possibility, and defer to the more general implementation. */
    def assessForMutualExclusion(proximityFilterStrategy: ProximityFilterStrategy)(rois: List[PostMergeRoi])(using Eq[FieldOfViewLike], AdmitsRoiIndex[PostMergeRoi]): 
        (List[UnidentifiableRoi], List[PostMergeRoi]) = 
        proximityFilterStrategy match {
            case UniversalProximityPermission => List() -> rois
            case strat: NontrivialProximityFilter => assessForMutualExclusion(strat)(rois)
        }

    def mergeRois(rois: List[MergerAssessedRoi])(using Semigroup[BoundingBox]): MergeResult = 
        rois match {
            case Nil => (List(), List(), List(), List())
            case _ => 
                val indexed = NonnegativeInt.indexed(rois)
                def incrementIndex: RoiIndex => RoiIndex = i => RoiIndex.unsafe(i.get + 1)
                val pool = indexed.map{ (r, i) => RoiIndex(i) -> r }.toMap
                given Ordering[RoiIndex] = summon[Order[RoiIndex]].toOrdering
                val initNewIndex = incrementIndex(rois.map(_.index).max)
                val ((allErrored, allSkipped, allMerged), _) = 
                    indexed.foldRight(((List.empty[MergeError], List.empty[IndexedDetectedSpot], List.empty[MergedRoiRecord]), initNewIndex)){
                        case (curr@(r, i), ((accErr, accSkip, accMerge), currIndex)) => 
                            considerOneMerge(pool)(currIndex, r) match {
                                case None => 
                                    // no merge action; simply eliminate the empty mergePartners collection
                                    (accErr, (r.index, r.roi) :: accSkip, accMerge) -> currIndex
                                case Some(Left(errors)) => 
                                    // error case
                                    ((curr, errors) :: accErr, accSkip, accMerge) -> currIndex
                                case Some(Right(rec)) =>
                                    // merge action
                                    (accErr, accSkip, rec :: accMerge) -> incrementIndex(currIndex)
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
                            original.roi.box,
                            mergedIndex, 
                        )
                    }
                    .sortBy(_.index)
                (allErrored, allSkipped, allContrib, allMerged)                
        }
    
    /** Do the merge for a single ROI record. */
    private[looptrace] def considerOneMerge(
        pool: Map[RoiIndex, MergerAssessedRoi]
    )(potentialNewIndex: RoiIndex, roi: MergerAssessedRoi)(using 
        Semigroup[BoundingBox]
    ): Option[Either[NonEmptyList[String], MergedRoiRecord]] = 
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
                        val newBox: BoundingBox = partners.map(_.roi.box).reduce
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

    private type MergeError = ((MergerAssessedRoi, NonnegativeInt), ErrorMessages)
    
    private[roi] type IndexedDetectedSpot = (RoiIndex, DetectedSpotRoi)

    private[roi] object IndexedDetectedSpot:
        given AdmitsRoiIndex[IndexedDetectedSpot] = AdmitsRoiIndex.instance(_._1)

        given admitsImagingContextForIndexedDetectedSpot(using forSpot: AdmitsImagingContext[DetectedSpotRoi]): AdmitsImagingContext[IndexedDetectedSpot] = 
            forSpot.contramap(_._2)
    end IndexedDetectedSpot

    private type MergeResult = (
        List[MergeError], // errors
        List[IndexedDetectedSpot], // non-participants in merge
        List[MergeContributorRoi], // contributors to merge
        List[MergedRoiRecord], // merge outputs
    )
end MergeAndSplitRoiTools