package at.ac.oeaw.imba.gerlich.looptrace

import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

/** A detected spot ROI not in a nucleus */
final case class NonNuclearRoi(
    index: RoiIndex, 
    context: ImagingContext,
    centroid: Centroid[Double], 
    box: BoundingBox
)

object NonNuclearRoi:
    def fromDetectedSpot(i: RoiIndex, roi: DetectedSpotRoi): NonNuclearRoi = 
        new NonNuclearRoi(
            i, 
            roi.context, 
            roi.centroid, 
            roi.box
        )
end NonNuclearRoi
