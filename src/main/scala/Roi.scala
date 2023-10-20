package at.ac.oeaw.imba.gerlich.looptrace

import upickle.default.*
import at.ac.oeaw.imba.gerlich.looptrace.space.{ Coordinate, CoordinateSequence, Point3D }

/** Type representing a detected fiducial bead region of interest (ROI) */
final case class DetectedRoi(index: RoiIndex, centroid: Point3D, isUsable: Boolean)

/** A region of interest (ROI) selected for use in some process */
sealed trait SelectedRoi {
    def index: RoiIndex
    def centroid: Point3D
}

/** A region of interest (ROI) selected for use in drift correction shifting */
final case class RoiForShifting(index: RoiIndex, centroid: Point3D) extends SelectedRoi

/** A region of interest (ROI) selected for use in drift correction accuracy assessment. */
final case class RoiForAccuracy(index: RoiIndex, centroid: Point3D) extends SelectedRoi

/** Helpers for working with indexed regions of interest (ROIs) */
object SelectedRoi:
    /** The key for the ROI's index in JSON representation */
    val indexKey: String = "index"
    /** The key for the ROI's point/centroid in JSON representation */
    val pointKey: String = "centroid"

    /** Serialise the index as a simple integer, and centroid as a simple array of Double, sequenced as requested. */
    def toJsonSimple(coordseq: CoordinateSequence)(roi: SelectedRoi)(using (Coordinate => ujson.Value)): ujson.Obj = 
        ujson.Obj(
            indexKey -> ujson.Num(roi.index.get), 
            pointKey -> ujson.Arr.from(Point3D.toList(coordseq)(roi.centroid).toList)
        )

    /** Serialise the index as a simple integer, and centroid as a simple array of Double, sequenced as requested. */
    def simpleJsonReadWriter[R <: SelectedRoi](coordseq: CoordinateSequence, build: (RoiIndex, Point3D) => R)(using (Coordinate => ujson.Value)): ReadWriter[R] = {
        readwriter[ujson.Value].bimap[R](
            toJsonSimple(coordseq), 
            json => {
                val rawIndex = NonnegativeInt.unsafe(json(indexKey).num.toInt)
                val idx = RoiIndex(NonnegativeInt.unsafe(rawIndex))
                val coords = json(pointKey).arr.map(_.num.toDouble)
                val pt = Point3D.fromList(coordseq)(coords.toList).fold(msg => throw new Exception(msg), identity)
                build(idx, pt)
            }
        )
    }

    /** JSON reader/writer for shifting-selected ROI, based on given coordinate sequencer */
    def simpleShiftingRW(coordseq: CoordinateSequence)(using (Coordinate) => ujson.Value): ReadWriter[RoiForShifting] = 
        simpleJsonReadWriter(coordseq, RoiForShifting.apply)
    
    /** JSON reader/writer for accuracy-selected ROI, based on given coordinate sequencer */
    def simpleAccuracyRW(coordseq: CoordinateSequence)(using (Coordinate) => ujson.Value): ReadWriter[RoiForAccuracy] = 
        simpleJsonReadWriter(coordseq, RoiForAccuracy.apply)

