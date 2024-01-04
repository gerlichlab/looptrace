package at.ac.oeaw.imba.gerlich.looptrace

import at.ac.oeaw.imba.gerlich.looptrace.space.Point3D

/** Representation of a single record from the regional barcode spots detection */
final case class RegionalBarcodeSpotRoi(
    index: RoiIndex, position: PositionName, time: FrameIndex, channel: Channel, centroid: Point3D, boundingBox: BoundingBox
    )
