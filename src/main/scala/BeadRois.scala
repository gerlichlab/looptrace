package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import scala.util.chaining.*

import cats.data.NonEmptyList
import cats.syntax.all.*
import upickle.default.*

import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnName
import at.ac.oeaw.imba.gerlich.gerlib.json.instances.all.{*, given}
import at.ac.oeaw.imba.gerlich.gerlib.json.syntax.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

import at.ac.oeaw.imba.gerlich.looptrace.space.{
    Coordinate, 
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
final case class FiducialBead(index: RoiIndex, centroid: Point3D) extends RoiLike

object FiducialBead:
    import fs2.data.csv.*
    
    given (
        decIdx: CellDecoder[RoiIndex], 
        decCenter: CsvRowDecoder[Centroid[Double], String],
    ) => CsvRowDecoder[FiducialBead, String] = 
        new:
            override def apply(row: RowF[Some, String]): DecoderResult[FiducialBead] = 
                val indexNel = ColumnName[RoiIndex]("beadIndex").from(row)
                val centerNel = decCenter(row)
                    .bimap(err => NonEmptyList.one(err.getMessage), _.asPoint)
                    .toValidated
                (indexNel, centerNel)
                    .mapN(FiducialBead.apply)
                    .toEither
                    .leftMap(messages => 
                        DecoderError(s"Problem(s) decoding bead: ${messages.mkString_("; ")}")
                    )
end FiducialBead

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

    given coord2Value: (Coordinate => ujson.Value):
        override def apply(c: Coordinate): ujson.Value = 
            given [C <: Coordinate: [C] =>> NotGiven[C =:= Coordinate]] => JsonValueWriter[C, ujson.Num] = 
                getPlainJsonValueWriter[Double, C, ujson.Num]
            c match {
                case x: XCoordinate => x.asJson
                case y: YCoordinate => y.asJson
                case z: ZCoordinate => z.asJson
            }

    /** Serialise the index as a simple integer, and centroid as a simple array of Double, sequenced as requested. */
    def toJsonSimple(roi: SelectedRoi): ujson.Obj = 
        import Point3D.given // for JsonValueWriter
        ujson.Obj(
            indexKey -> ujson.Num(roi.index.get), 
            pointKey -> roi.centroid.asJson,
        )

    /** Serialise the index as a simple integer, and centroid as a simple array of Double, sequenced as requested. */
    private def simpleJsonReadWriter[R <: SelectedRoi : [R] =>> NotGiven[R =:= SelectedRoi]](build: (RoiIndex, Point3D) => R): ReadWriter[R] = 
        import Point3D.given // for Reader[Point3D]
        readwriter[ujson.Value].bimap[R](
            toJsonSimple, 
            json => {
                val i = json(indexKey).num.toInt.pipe(RoiIndex.unsafe)
                val p = json(pointKey).pipe(read[Point3D](_, false))
                build(i, p)
            }
        )

    /** JSON reader/writer for shifting-selected ROI, based on given coordinate sequencer */
    def simpleShiftingRW: ReadWriter[RoiForShifting] = simpleJsonReadWriter(RoiForShifting.apply)
    
    /** JSON reader/writer for accuracy-selected ROI, based on given coordinate sequencer */
    def simpleAccuracyRW: ReadWriter[RoiForAccuracy] = simpleJsonReadWriter(RoiForAccuracy.apply)

