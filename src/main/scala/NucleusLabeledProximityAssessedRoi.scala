package at.ac.oeaw.imba.gerlich.looptrace

import cats.Id
import cats.data.NonEmptyList
import cats.syntax.all.*
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.collections.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oewa.imba.gerlich.looptrace.RowIndexAdmission
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given

/** A ROI already assessed for nuclear attribution and proximity to other ROIs */
final case class NucleusLabeledProximityAssessedRoi private(
    index: RoiIndex, 
    roi: DetectedSpotRoi, 
    nucleus: NuclearDesignation,
    tooCloseNeighbors: Set[RoiIndex],
    mergeNeighbors: Set[RoiIndex],
    analyticalGroupingPartners: Set[RoiIndex],
):
    def centroid: Centroid[Double] = roi.centroid
    def context: ImagingContext = roi.spot.context
    def dropNeighbors: NucleusLabelAttemptedRoi = NucleusLabelAttemptedRoi(roi, nucleus)

/** Tools for working with ROIs already assessed for nuclear attribution and proximity to other ROIs */
object NucleusLabeledProximityAssessedRoi:

    def build(
        index: RoiIndex, 
        roi: DetectedSpotRoi, 
        nucleus: NuclearDesignation, 
        tooClose: Set[RoiIndex], 
        merge: Set[RoiIndex],
        groupForAnalysis: Set[RoiIndex],
    ): Either[NonEmptyList[String], NucleusLabeledProximityAssessedRoi] = 
        val selfTooCloseNel = tooClose.excludes(index)
            .validatedNel(s"An ROI cannot be too close to itself (index ${index.show_})", ())
        val selfMergeNel = merge.excludes(index)
            .validatedNel(s"An ROI cannot be merged with itself (index ${index.show_})", ())
        val selfGroupNel = groupForAnalysis.excludes(index)
            .validatedNel(s"An ROI cannot be grouped with itself (index ${index.show_})", ())
        val closeMergeDisjointNel = 
            val overlap = tooClose & merge
            overlap.isEmpty.validatedNel(s"Overlap between too-close ROIs and ROIs to merge: ${overlap}", ())
        val closeGroupDisjointNel = 
            val overlap = tooClose & groupForAnalysis
            overlap.isEmpty.validatedNel(s"Overlap between too-close ROIs and ROIs to group analytically: ${overlap}", ())
        val mergeGroupDisjointNel = 
            val overlap = merge & groupForAnalysis
            overlap.isEmpty.validatedNel(s"Overlap between ROIs to merge and ROIs to group analytically: ${overlap}", ())

        (selfTooCloseNel, selfMergeNel, selfGroupNel, closeMergeDisjointNel, closeGroupDisjointNel, mergeGroupDisjointNel)
            .tupled
            .map{
                Function.const{
                    singleton(index, roi, nucleus).copy(
                        tooCloseNeighbors = tooClose, 
                        mergeNeighbors = merge,
                        analyticalGroupingPartners = groupForAnalysis
                    )
                }
            }
            .toEither

    def singleton(
        index: RoiIndex, 
        roi: DetectedSpotRoi, 
        nucleus: NuclearDesignation,
    ): NucleusLabeledProximityAssessedRoi = 
        new NucleusLabeledProximityAssessedRoi(index, roi, nucleus, Set(), Set(), Set())

    given ProximityExclusionAssessedRoiLike[NucleusLabeledProximityAssessedRoi] with
        override def getRoiIndex = _.index
        override def getTooCloseNeighbors = _.tooCloseNeighbors

    given ProximityMergeAssessedRoiLike[NucleusLabeledProximityAssessedRoi] with 
        override def getRoiIndex = _.index
        override def getMergeNeighbors = _.mergeNeighbors

    given RowIndexAdmission[NucleusLabeledProximityAssessedRoi, Id] = 
        RowIndexAdmission.intoIdentity(_.index.get)
end NucleusLabeledProximityAssessedRoi