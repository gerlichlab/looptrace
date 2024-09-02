package at.ac.oeaw.imba.gerlich.looptrace
package roi

import cats.syntax.all.*

import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

/** A ROI that's merged with one or more others on account of proximity. */
private[looptrace] final case class MergeContributorRoi(
    index: RoiIndex, 
    context: ImagingContext,
    centroid: Centroid[Double],
    box: BoundingBox, 
    mergeIndex: RoiIndex
)

object MergeContributorRoi:
    private[looptrace] def apply(index: RoiIndex, spot: DetectedSpotRoi, mergeIndex: RoiIndex): Either[String, MergeContributorRoi] = 
        if index === mergeIndex 
        then s"Merge contributor ROI index is the same as the merge result index: ${index} === ${mergeIndex}".asLeft
        else MergeContributorRoi(
            index,
            spot.context, 
            spot.centroid, 
            spot.box, 
            mergeIndex,
        ).asRight
