package at.ac.oeaw.imba.gerlich.looptrace

import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.roi.DetectedSpot
import at.ac.oeaw.imba.gerlich.gerlib.roi.measurement.*

import at.ac.oeaw.imba.gerlich.looptrace.space.*

/** Bundle of imaging context, ROI centroid, size, mean pixel intensity, and a 3D bounding box */
final case class DetectedSpotRoi(
    context: ImagingContext,
    center: Point3D,
    area: Area,
    intensity: MeanIntensity, 
    box: BoundingBox,
)

/** Helpers for working with ROIs which come from initial spot detection */
object DetectedSpotRoi:
    def apply(spot: DetectedSpot[Double], box: BoundingBox): DetectedSpotRoi = 
        DetectedSpotRoi(
            spot.context, 
            Point3D(spot.centerX, spot.centerY, spot.centerZ), 
            spot.area, 
            spot.intensity, 
            box
        )
