package at.ac.oeaw.imba.gerlich.looptrace
package roi

import cats.data.NonEmptySet
import cats.syntax.all.*

import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.nonnegativeInt.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

/** A ROI that's merged with one or more others on account of proximity. */
private[looptrace] final case class MergeContributorRoi(
    index: RoiIndex, 
    context: ImagingContext,
    centroid: Centroid[Double],
    box: BoundingBox, 
    mergeIndices: NonEmptySet[RoiIndex],
)

object MergeContributorRoi:
    private[looptrace] def apply(index: RoiIndex, spot: DetectedSpotRoi, mergeIndices: NonEmptySet[RoiIndex]): Either[String, MergeContributorRoi] = 
        if mergeIndices `contains` index
        then s"Merge contributor ID is among its own merge result IDs: ${index.show_} in ${mergeIndices}".asLeft
        else MergeContributorRoi(
            index,
            spot.context, 
            spot.centroid, 
            spot.box, 
            mergeIndices,
        ).asRight

    given AdmitsRoiIndex[MergeContributorRoi] = AdmitsRoiIndex.instance(_.index)

    given AdmitsImagingContext[MergeContributorRoi] = AdmitsImagingContext.instance(_.context)
end MergeContributorRoi
