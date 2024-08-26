package at.ac.oeaw.imba.gerlich.looptrace

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.gerlib.roi.DetectedSpot
import at.ac.oeaw.imba.gerlich.gerlib.roi.measurement.Area
import at.ac.oeaw.imba.gerlich.gerlib.roi.measurement.MeanIntensity

import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

/** Add designation of nucleus to the data associated with a FISH spot ROI. */
final case class NucleusLabelAttemptedRoi(
    spot: DetectedSpot[Double],
    box: BoundingBox, 
    nucleus: NuclearDesignation
):
    def context: ImagingContext = spot.context
    def centroid: Centroid[Double] = spot.centroid
    def toDetectedSpotRoi: DetectedSpotRoi = DetectedSpotRoi(spot, box)

/** Helpers for working with detected ROIs for which nuclear attribution has been attempted */
object NucleusLabelAttemptedRoi:
    /** Simply add the nucleus designation to the other information from the detected ROI. */
    def apply(roi: DetectedSpotRoi, nucleus: NuclearDesignation): NucleusLabelAttemptedRoi = 
        NucleusLabelAttemptedRoi(
            roi.spot, 
            roi.box,
            nucleus,
        )
end NucleusLabelAttemptedRoi