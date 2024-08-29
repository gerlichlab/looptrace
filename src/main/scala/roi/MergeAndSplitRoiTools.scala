package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.{ NonEmptyList, NonEmptySet, ValidatedNel }
import cats.syntax.all.*
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.cell.{ NuclearDesignation, OutsideNucleus }
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.geometry.syntax.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.space.{ BoundingBox, Point3D }
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt

/** Tools for merging ROIs */
object MergeAndSplitRoiTools:
    private type Numbered[A] = (A, NonnegativeInt)

    private type MergeResult = (
        List[(Numbered[MergerAssessedRoi], NonEmptyList[String])], // errors
        List[Numbered[MergerAssessedRoi]], // non-participants in merge
        List[MergeContributorRoi], // merge inputs
        List[MergedRoiRecord], // merge outputs
    )

    /** A record of an ROI after the merge process has been considered and done. */
    private[looptrace] final case class MergedRoiRecord private[MergeAndSplitRoiTools](
        index: RoiIndex, 
        context: ImagingContext, // must be identical among all merge partners
        centroid: Centroid[Double], // averaged over merged partners
        box: BoundingBox, 
        contributors: NonEmptySet[RoiIndex], 
    )

    def mergeRois(rois: List[MergerAssessedRoi])(using Order[NuclearDesignation], Semigroup[BoundingBox]): MergeResult = 
        val initAcc: MergeResult = (List(), List(), List(), List())
        rois match {
            case Nil => initAcc
            case _ => 
                val indexed = NonnegativeInt.indexed(rois)
                def incrementIndex: RoiIndex => RoiIndex = i => RoiIndex.unsafe(i.get + 1)
                val pool = indexed.map{ (r, i) => RoiIndex(i) -> r }.toMap
                given Ordering[RoiIndex] = summon[Order[RoiIndex]].toOrdering
                val initNewIndex = incrementIndex(rois.map(_.index).max)
                indexed.foldRight(initAcc -> initNewIndex){ case (curr@(r, i), ((accErr, accSkip, accContrib, accMerge), currIndex)) => 
                    doOneMerge(pool)(currIndex, r) match {
                        case None => 
                            // no merge action
                            (accErr, curr :: accSkip, accContrib, accMerge) -> currIndex
                        case Some(Left(errors)) => 
                            // error case
                            ((curr, errors) :: accErr, accSkip, accContrib, accMerge) -> currIndex
                        case Some(Right(rec)) => 
                            // merge action
                            (accErr, accSkip, accContrib, rec :: accMerge) -> incrementIndex(currIndex)
                    }
                }._1
        }

    /** Do the merge for a single ROI record. */
    private[looptrace] def doOneMerge(
        pool: Map[RoiIndex, MergerAssessedRoi]
    )(potentialNewIndex: RoiIndex, roi: MergerAssessedRoi)(using 
        Order[NuclearDesignation], 
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
