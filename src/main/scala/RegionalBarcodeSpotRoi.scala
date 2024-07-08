package at.ac.oeaw.imba.gerlich.looptrace

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.looptrace.space.Point3D

/** Representation of a single record from the regional barcode spots detection */
final case class RegionalBarcodeSpotRoi(
    index: RoiIndex, position: PositionName, region: RegionId, channel: Channel, centroid: Point3D, boundingBox: BoundingBox
    ):
    final def time: ImagingTimepoint = region.get
