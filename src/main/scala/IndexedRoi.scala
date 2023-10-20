package at.ac.oeaw.imba.gerlich.looptrace

import at.ac.oeaw.imba.gerlich.looptrace.space.{ Coordinate, CoordinateSequence, Point3D }

/** Supertype for regions of interest (ROIs) */
sealed trait IndexedRoi {
    def index: RoiIndex
    def centroid: Point3D
}

/** Type representing a detected fiducial bead region of interest (ROI) */
final case class DetectedRoi(index: RoiIndex, centroid: Point3D, isUsable: Boolean) extends IndexedRoi

/** A region of interest (ROI) selected for use in some process */
sealed trait SelectedRoi extends IndexedRoi

/** A region of interest (ROI) selected for use in drift correction shifting */
final case class RoiForShifting(index: RoiIndex, centroid: Point3D) extends SelectedRoi

/** A region of interest (ROI) selected for use in drift correction accuracy assessment. */
final case class RoiForAccuracy(index: RoiIndex, centroid: Point3D) extends SelectedRoi

/** Helpers for working with indexed regions of interest (ROIs) */
object IndexedRoi {
    def toJsonSimple(coordseq: CoordinateSequence)(roi: IndexedRoi)(using (Coordinate => ujson.Value)): ujson.Obj = 
        ujson.Obj(
            "index" -> ujson.Num(roi.index.get), 
            "centroid" -> ujson.Arr.from(Point3D.toList(coordseq)(roi.centroid).toList)
        )
    
    def toJsonSimple(coordseq: CoordinateSequence)(rois: Iterable[IndexedRoi])(using (Coordinate => ujson.Value)): ujson.Arr = {
        val serialise1 = toJsonSimple(coordseq)(_: IndexedRoi)
        ujson.Arr(rois map serialise1)
    }
}
