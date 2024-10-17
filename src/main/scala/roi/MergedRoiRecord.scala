package at.ac.oeaw.imba.gerlich.looptrace
package roi

import cats.data.NonEmptySet

import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.IndexedDetectedSpot

/** A record of an ROI after the merge process has been considered and done. */
private[looptrace] final case class MergedRoiRecord(
    index: RoiIndex, 
    context: ImagingContext, // must be identical among all merge partners
    centroid: Centroid[Double], // averaged over merged partners
    box: BoundingBox, 
    contributors: NonEmptySet[RoiIndex], 
)

/** Helpers for working with merged ROI records */
private[looptrace] object MergedRoiRecord:
    /** Alternate constructor based on adding an index and contributor indices to a detected spot ROI */
    def apply(spot: IndexedDetectedSpot, contributors: NonEmptySet[RoiIndex]): MergedRoiRecord = 
        new MergedRoiRecord(
            spot.index, 
            spot.context, 
            spot.centroid, 
            spot.box, 
            contributors
        )

    given AdmitsRoiIndex[MergedRoiRecord] = AdmitsRoiIndex.instance(_.index)

    given AdmitsImagingContext[MergedRoiRecord] = AdmitsImagingContext.instance(_.context)
end MergedRoiRecord