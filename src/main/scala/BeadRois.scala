package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import upickle.default.*

import at.ac.oeaw.imba.gerlich.gerlib.json.instances.all.{*, given}
import at.ac.oeaw.imba.gerlich.gerlib.json.syntax.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

import at.ac.oeaw.imba.gerlich.looptrace.space.{
    Coordinate, 
    CoordinateSequence, 
    Point3D, 
    XCoordinate, 
    YCoordinate, 
    ZCoordinate,
}
import at.ac.oeaw.imba.gerlich.gerlib.json.JsonValueWriter

/** A type wrapper around a string with which to represent the reason(s) why a bead ROI is unusable */
final case class RoiFailCode(get: String):
    def indicatesFailure: Boolean = get.nonEmpty
    def indicatesSuccess: Boolean = !indicatesFailure
end RoiFailCode

/** Helpers for working with {@code RoiFailCode} values */
object RoiFailCode:
    def success = RoiFailCode("")

/** An entity which has a {@code RoiIndex} and a centroid */
sealed trait RoiLike:
    def index: RoiIndex
    def centroid: Point3D

/** Type representing a detected fiducial bead region of interest (ROI) */
final case class FiducialBeadRoi(index: RoiIndex, centroid: Point3D, failCode: RoiFailCode) extends RoiLike:
    def isUsable: Boolean = failCode.indicatesSuccess

/** A region of interest (ROI) selected for use in some process */
sealed trait SelectedRoi extends RoiLike {
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
    private val indexKey: String = "index"
    /** The key for the ROI's point/centroid in JSON representation */
    private val pointKey: String = "centroid"

    given coord2Value: (Coordinate => ujson.Value) with
        override def apply(c: Coordinate): ujson.Value = 
            given writerForCoordinate[C <: Coordinate: [C] =>> NotGiven[C =:= Coordinate]]: JsonValueWriter[C, ujson.Num] = 
                getPlainJsonValueWriter[Double, C, ujson.Num]
            c match {
                case x: XCoordinate => x.asJson
                case y: YCoordinate => y.asJson
                case z: ZCoordinate => z.asJson
            }

    /** Serialise the index as a simple integer, and centroid as a simple array of Double, sequenced as requested. */
    def toJsonSimple(coordseq: CoordinateSequence)(roi: SelectedRoi): ujson.Obj = 
        ujson.Obj(
            indexKey -> ujson.Num(roi.index.get), 
            pointKey -> ujson.Arr.from(Point3D.toList(coordseq)(roi.centroid).toList)
        )

    /** Serialise the index as a simple integer, and centroid as a simple array of Double, sequenced as requested. */
    private def simpleJsonReadWriter[R <: SelectedRoi : [R] =>> NotGiven[R =:= SelectedRoi]](
        coordseq: CoordinateSequence, build: (RoiIndex, Point3D) => R
        ): ReadWriter[R] = {
        readwriter[ujson.Value].bimap[R](
            toJsonSimple(coordseq), 
            json => {
                val rawIndex = NonnegativeInt.unsafe(json(indexKey).num.toInt)
                val idx = RoiIndex.unsafe(rawIndex)
                val coords = json(pointKey).arr.map(_.num.toDouble)
                val pt = Point3D.fromList(coordseq)(coords.toList).fold(msg => throw new Exception(msg), identity)
                build(idx, pt)
            }
        )
    }

    /** JSON reader/writer for shifting-selected ROI, based on given coordinate sequencer */
    def simpleShiftingRW(coordseq: CoordinateSequence): ReadWriter[RoiForShifting] = 
        simpleJsonReadWriter(coordseq, RoiForShifting.apply)
    
    /** JSON reader/writer for accuracy-selected ROI, based on given coordinate sequencer */
    def simpleAccuracyRW(coordseq: CoordinateSequence): ReadWriter[RoiForAccuracy] = 
        simpleJsonReadWriter(coordseq, RoiForAccuracy.apply)

