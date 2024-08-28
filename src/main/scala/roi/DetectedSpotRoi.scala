package at.ac.oeaw.imba.gerlich.looptrace

import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.roi.DetectedSpot
import at.ac.oeaw.imba.gerlich.gerlib.roi.measurement.*

import at.ac.oeaw.imba.gerlich.looptrace.space.*

/** Bundle of imaging context, ROI centroid, size, mean pixel intensity, and a 3D bounding box */
final case class DetectedSpotRoi(
    spot: DetectedSpot[Double],
    box: BoundingBox,
):
    def centroid: Centroid[Double] = spot.centroid
    def context: ImagingContext = spot.context