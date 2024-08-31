package at.ac.oeaw.imba.gerlich.looptrace
package roi

import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.looptrace.space.{ BoundingBox, Point3D }

/** Representation of a single record from the regional barcode spots detection */
final case class RegionalBarcodeSpotRoi(
    index: RoiIndex, position: PositionName, region: RegionId, channel: ImagingChannel, centroid: Point3D, boundingBox: BoundingBox
    ):
    final def time: ImagingTimepoint = region.get
