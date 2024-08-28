package at.ac.oeaw.imba.gerlich.looptrace

import at.ac.oeaw.imba.gerlich.gerlib.cell.NucleusNumber
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

/** A ROI that's merged with one or more others on account of proximity. */
final case class MergeContributorRoi(
    index: RoiIndex, 
    context: ImagingContext,
    centroid: Centroid[Double],
    box: BoundingBox, 
    nucleus: NucleusNumber,
    mergeIndex: RoiIndex
)
