package at.ac.oeaw.imba.gerlich.looptrace

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.gerlib.roi.measurement.Area
import at.ac.oeaw.imba.gerlich.gerlib.roi.measurement.MeanIntensity

import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

/** Add designation of nucleus to the data associated with a FISH spot ROI. */
final case class NucleusLabelAttemptedRoi(
    context: ImagingContext, 
    center: Centroid[Double], 
    area: Area, 
    intensity: MeanIntensity, 
    box: BoundingBox, 
    nucleus: NuclearDesignation
)
end NucleusLabelAttemptedRoi

/** Helpers for working with detected ROIs for which nuclear attribution has been attempted */
object NucleusLabelAttemptedRoi:
    /** Simply add the nucleus designation to the other information from the detected ROI. */
    def apply(detected: DetectedSpotRoi, nucleus: NuclearDesignation): NucleusLabelAttemptedRoi = 
        NucleusLabelAttemptedRoi(
            detected.context, 
            detected.center, 
            detected.area, 
            detected.intensity, 
            detected.box, 
            nucleus,
        )
