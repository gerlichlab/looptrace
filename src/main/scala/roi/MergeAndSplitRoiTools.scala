package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.NonEmptyList
import cats.syntax.all.*
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.geometry.syntax.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.space.{ BoundingBox, Point3D }
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt

/** Tools for merging ROIs */
object MergeAndSplitRoiTools:
    private type Numbered[A] = (A, NonnegativeInt)

    private type MergeResult = (
        List[(Numbered[NucleusLabeledProximityAssessedRoi], NonEmptyList[String])], 
        List[Numbered[NucleusLabeledProximityAssessedRoi]], 
        List[MergedRoiRecord],
    )

    // private type SplitResult = (
    //     List[],
    //     List[],
    //     List[MergedRoiRecord]
    // )

    def mergeRois(rois: List[NucleusLabeledProximityAssessedRoi])(using Order[NuclearDesignation], Monoid[BoundingBox]): MergeResult = 
        val initAcc: MergeResult = (List(), List(), List())
        rois match {
            case Nil => initAcc
            case _ => 
                val indexed = NonnegativeInt.indexed(rois)
                def incrementIndex: RoiIndex => RoiIndex = i => RoiIndex.unsafe(i.get + 1)
                val pool = indexed.map{ (r, i) => RoiIndex(i) -> r }.toMap
                given Ordering[RoiIndex] = summon[Order[RoiIndex]].toOrdering
                val initNewIndex = incrementIndex(rois.map(_.index).max)
                indexed.foldRight(initAcc -> initNewIndex){ case (curr@(r, i), ((accErr, accSkip, accMerge), currIndex)) => 
                    doOneMerge(pool)(currIndex, r) match {
                        case None => (accErr, curr :: accSkip, accMerge) -> currIndex
                        case Some(Left(errors)) => ((curr, errors) :: accErr, accSkip, accMerge) -> currIndex
                        case Some(Right(rec)) => (accErr, accSkip, rec :: accMerge) -> incrementIndex(currIndex)
                    }
                }._1
        }

    // def splitRois(rois: List[NucleusLabeledProximityAssessedRoi | MergedRoiRecord]): (

    // )

    /** Do the merge for a single ROI record. */
    private[looptrace] def doOneMerge(
        pool: Map[RoiIndex, NucleusLabeledProximityAssessedRoi]
    )(potentialNewIndex: RoiIndex, roi: NucleusLabeledProximityAssessedRoi)(using 
        Order[NuclearDesignation], 
        Monoid[BoundingBox]
    ): Option[Either[NonEmptyList[String], MergedRoiRecord]] = 
        roi.mergeNeighbors.toList.toNel.map(
            _.traverse{ i => 
                pool.get(i)
                    .toRight(s"Missing ROI index: ${i.get}")
                    .toValidatedNel
            }
            .toEither
            .flatMap{ partners => 
                val nucs = partners.map(_.nucleus).toNes
                val contexts = partners.map(_.context).toNes
                val nucleusNel = (nucs.size === 1).validatedNel(
                    s"${nucs.size} unique nuclei designations (not just 1) in ROI group to merge", 
                    partners.head.nucleus
                )
                val contextNel = (contexts.size === 1).validatedNel(
                    s"${contexts.size} unique imaging context (not just 1) in ROI group to merge", 
                    partners.head.context
                )
                (nucleusNel, contextNel).mapN((nucleus, context) => 
                    val newCenter: Point3D = partners.map(_.centroid.asPoint).centroid
                    val newBox: BoundingBox = partners.map(_.roi.box).combineAll
                    val result = MergedRoiRecord(
                        potentialNewIndex, 
                        context, 
                        Centroid.fromPoint(newCenter), 
                        newBox, 
                        nucleus,
                        partners.map(_.index).toNes,
                    )
                    result
                )
                .toEither
            }
        )
