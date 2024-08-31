package at.ac.oeaw.imba.gerlich.looptrace
package roi

import cats.*
import cats.data.{ NonEmptyList, NonEmptySet, ValidatedNel }
import cats.syntax.all.*
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.geometry.syntax.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.space.{ BoundingBox, Point3D }
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.ProximityFilterStrategy
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.UniversalProximityPermission
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.UniversalProximityProhibition
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.SelectiveProximityPermission
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.SelectiveProximityProhibition

/** Tools for merging ROIs */
object MergeAndSplitRoiTools:
    def assessForMerge(rois: List[DetectedSpotRoi]): List[MergerAssessedRoi] = ???

    def assessForMutualExclusion(proximityFilterStrategy: ProximityFilterStrategy)(rois: List[IndexedSpot | MergedRoiRecord]): 
        (List[UnidentifiableRoi], List[IndexedSpot | MergedRoiRecord]) = 
        proximityFilterStrategy match {
            case UniversalProximityPermission => List() -> rois
            case UniversalProximityProhibition(minSpotSeparation) => ???
            case SelectiveProximityPermission(minSpotSeparation, grouping) => ???
            case SelectiveProximityProhibition(minSpotSeparation, grouping) => ???
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
                    indexed.foldRight(((List.empty[MergeError], List.empty[IndexedSpot], List.empty[MergedRoiRecord]), initNewIndex)){
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

    /** A ROI that's merged with one or more others on account of proximity. */
    private[looptrace] final case class MergeContributorRoi(
        index: RoiIndex, 
        context: ImagingContext,
        centroid: Centroid[Double],
        box: BoundingBox, 
        mergeIndex: RoiIndex
    )

    /** A record of an ROI after the merge process has been considered and done. */
    private[looptrace] final case class MergedRoiRecord(
        index: RoiIndex, 
        context: ImagingContext, // must be identical among all merge partners
        centroid: Centroid[Double], // averaged over merged partners
        box: BoundingBox, 
        contributors: NonEmptySet[RoiIndex], 
    )

    private type MergeError = ((MergerAssessedRoi, NonnegativeInt), ErrorMessages)
    
    private type IndexedSpot = (RoiIndex, DetectedSpotRoi)

    private type MergeResult = (
        List[MergeError], // errors
        List[IndexedSpot], // non-participants in merge
        List[MergeContributorRoi], // contributors to merge
        List[MergedRoiRecord], // merge outputs
    )
