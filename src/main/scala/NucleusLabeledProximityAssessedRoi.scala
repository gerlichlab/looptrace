package at.ac.oeaw.imba.gerlich.looptrace

import cats.data.NonEmptyList
import cats.syntax.all.*
import mouse.boolean.*
import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation

/** A ROI already assessed for nuclear attribution and proximity to other ROIs */
final case class NucleusLabeledProximityAssessedRoi private(
    index: RoiIndex, 
    roi: DetectedSpotRoi, 
    nucleus: NuclearDesignation,
    tooCloseNeighbors: Set[RoiIndex],
    mergeNeighbors: Set[RoiIndex],
):
    def dropNeighbors: NucleusLabelAttemptedRoi = NucleusLabelAttemptedRoi(roi, nucleus)

/** Tools for working with ROIs already assessed for nuclear attribution and proximity to other ROIs */
object NucleusLabeledProximityAssessedRoi:

    def build(
        index: RoiIndex, 
        roi: DetectedSpotRoi, 
        nucleus: NuclearDesignation, 
        tooClose: Set[RoiIndex], 
        merge: Set[RoiIndex],
    ): Either[NonEmptyList[String], NucleusLabeledProximityAssessedRoi] = 
        val selfTooCloseNel = tooClose.contains(index).validatedNel("", ())
        val selfMergeNel = merge.contains(index).validatedNel("", ())
        val closeMergeDisjointNel = 
            val overlap = tooClose & merge
            overlap.isEmpty.validatedNel("", ())
        (selfTooCloseNel, selfMergeNel, closeMergeDisjointNel)
            .tupled
            .map{ Function.const{ singleton(index, roi, nucleus) } }
            .toEither

    def singleton(
        index: RoiIndex, 
        roi: DetectedSpotRoi, 
        nucleus: NuclearDesignation,
    ): NucleusLabeledProximityAssessedRoi = 
        new NucleusLabeledProximityAssessedRoi(index, roi, nucleus, Set(), Set())

    given ProximityExclusionAssessedRoiLike[NucleusLabeledProximityAssessedRoi] with
        override def getRoiIndex = _.index
        override def getTooCloseNeighbors = _.tooCloseNeighbors

    given ProximityMergeAssessedRoiLike[NucleusLabeledProximityAssessedRoi] with 
        override def getRoiIndex = _.index
        override def getMergeNeighbors = _.mergeNeighbors
end NucleusLabeledProximityAssessedRoi