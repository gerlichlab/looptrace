package at.ac.oeaw.imba.gerlich.looptrace
package roi

import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.looptrace.space.{ BoundingBox, Point3D }

/** Representation of a single record from the regional barcode spots detection */
final case class RegionalBarcodeSpotRoi(
    index: RoiIndex, 
    position: PositionName, 
    region: RegionId, 
    channel: ImagingChannel, 
    centroid: Point3D, 
    boundingBox: BoundingBox,
    contributors: Set[RoiIndex],
):
    final def time: ImagingTimepoint = region.get
end RegionalBarcodeSpotRoi

object RegionalBarcodeSpotRoi:
    given AdmitsImagingContext[RegionalBarcodeSpotRoi] = 
        AdmitsImagingContext.instance{ roi => 
            ImagingContext(roi.position, roi.time, roi.channel)
        }
end RegionalBarcodeSpotRoi
