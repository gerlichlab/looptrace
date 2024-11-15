package at.ac.oeaw.imba.gerlich.looptrace
package roi

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
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.IndexedDetectedSpot
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

/** A ROI already assessed for nuclear attribution and proximity to other ROIs */
final case class MergerAssessedRoi private(
    spot: IndexedDetectedSpot, 
    mergeNeighbors: Set[RoiIndex],
):
    def box: BoundingBox = spot.box
    def centroid: Centroid[Double] = spot.centroid
    def context: ImagingContext = spot.context
    def index: RoiIndex = spot.index

/** Tools for working with ROIs already assessed for nuclear attribution and proximity to other ROIs */
object MergerAssessedRoi:

    def build(
        index: RoiIndex, 
        context: ImagingContext, 
        centroid: Centroid[Double], 
        box: BoundingBox,
        merge: Set[RoiIndex],
    ): Either[String, MergerAssessedRoi] = 
        val spot = IndexedDetectedSpot(index, context, centroid, box)
        build(spot, merge)

    def build(
        spot: IndexedDetectedSpot, 
        merge: Set[RoiIndex],
    ): Either[String, MergerAssessedRoi] = 
        merge.excludes(spot.index).either(
            s"An ROI cannot be merged with itself (index ${spot.index.show_})",
            singleton(spot).copy(mergeNeighbors = merge)
        )

    def singleton(spot: IndexedDetectedSpot): MergerAssessedRoi = 
        new MergerAssessedRoi(spot, Set())

    given RowIndexAdmission[MergerAssessedRoi, Id] = 
        RowIndexAdmission.intoIdentity(_.index.get)

    given AdmitsRoiIndex[MergerAssessedRoi] = AdmitsRoiIndex.instance(_.index)

    given AdmitsImagingContext[MergerAssessedRoi] = AdmitsImagingContext.instance(_.context)
end MergerAssessedRoi