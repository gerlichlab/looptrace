package at.ac.oeaw.imba.gerlich.looptrace

import at.ac.oeaw.imba.gerlich.gerlib.cell.NucleusNumber
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

final case class NuclearSingleRoi(
    index: RoiIndex, 
    context: ImagingContext,
    centroid: Centroid[Double], 
    box: BoundingBox, 
    nucleus: NucleusNumber
)

object NuclearSingleRoi:
    def fromDetectedSpot(i: RoiIndex, roi: DetectedSpotRoi, nucleus: NucleusNumber): NuclearSingleRoi = 
        new NuclearSingleRoi(i, roi.context, roi.centroid, roi.box, nucleus)
end NuclearSingleRoi
