package at.ac.oeaw.imba.gerlich.looptrace

import cats.Id
import cats.data.NonEmptyList
import cats.syntax.all.*
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.collections.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oewa.imba.gerlich.looptrace.RowIndexAdmission
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given

/** A ROI already assessed for nuclear attribution and proximity to other ROIs */
final case class MergerAssessedRoi private(
    index: RoiIndex, 
    roi: DetectedSpotRoi, 
    mergeNeighbors: Set[RoiIndex],
):
    def centroid: Centroid[Double] = roi.centroid
    def context: ImagingContext = roi.context

/** Tools for working with ROIs already assessed for nuclear attribution and proximity to other ROIs */
object MergerAssessedRoi:

    def build(
        index: RoiIndex, 
        roi: DetectedSpotRoi, 
        merge: Set[RoiIndex],
    ): Either[String, MergerAssessedRoi] = 
        merge.excludes(index).either(
            s"An ROI cannot be merged with itself (index ${index.show_})",
            singleton(index, roi).copy(mergeNeighbors = merge)
        )

    def singleton(
        index: RoiIndex, 
        roi: DetectedSpotRoi, 
    ): MergerAssessedRoi = 
        new MergerAssessedRoi(index, roi, Set())

    given RowIndexAdmission[MergerAssessedRoi, Id] = 
        RowIndexAdmission.intoIdentity(_.index.get)
end MergerAssessedRoi