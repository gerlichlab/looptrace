package at.ac.oeaw.imba.gerlich.looptrace
package csv
package instances

import scala.util.{ NotGiven, Try }
import cats.data.{ NonEmptyList, ValidatedNel }
import cats.syntax.all.*
import fs2.data.csv.{ CellDecoder, CsvRowDecoder, DecoderError, DecoderResult, RowF }

import at.ac.oeaw.imba.gerlich.gerlib.geometry.Coordinate
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.liftToCellDecoder
import at.ac.oeaw.imba.gerlich.looptrace.space.{
    BoundingBox,
    Point3D,
    XCoordinate, 
    YCoordinate, 
    ZCoordinate,
}

/** IO-related typeclass instances for spatial data types */
trait SpatialCsvInstances:

    /** Decoder for bounding box records from CSV with new headers */
    def newCsvRowDecoderForBoundingBox: CsvRowDecoder[BoundingBox, String] = new:
        /** Parse each interval endpoint, then make assemble the intervals. */
        override def apply(row: CsvRow): DecoderResult[BoundingBox] = 
            val zNels = row.getPair[ZCoordinate]("zMin", "zMax")
            val yNels = row.getPair[YCoordinate]("yMin", "yMax")
            val xNels = row.getPair[XCoordinate]("xMin", "xMax")
            buildBox(zNels, yNels, xNels).leftMap(row.buildDecoderError)

    /** Decoder for bounding box records from CSV with old headers */
    def oldCsvRowDecoderForBoundingBox: CsvRowDecoder[BoundingBox, String] = new:
        /** Parse each interval endpoint, then make assemble the intervals. */
        override def apply(row: CsvRow): DecoderResult[BoundingBox] = 
            val zNels = row.getPair[ZCoordinate]("z_min", "z_max")
            val yNels = row.getPair[YCoordinate]("y_min", "y_max")
            val xNels = row.getPair[XCoordinate]("x_min", "x_max")
            buildBox(zNels, yNels, xNels).leftMap(row.buildDecoderError)

    /** Attempt to parse the given string as a double. */
    private def readDouble: String => Either[String, Double] = s => 
        Try{ s.toDouble }
            .toEither
            .leftMap(Function.const{ s"Cannot read as double: $s" })
    
    /** Create a [[fs2.data.csv.CellDecoder]] from this trait' double parser. */
    private given CellDecoder[Double] = liftToCellDecoder(readDouble)

    private type CsvRow = RowF[Some, String]

    private type MaybeEndpoints[C <: Coordinate[Double]] = (ValidatedNel[String, C], ValidatedNel[String, C])

    extension (row: CsvRow)
        private def get[C <: Coordinate[Double] : CellDecoder : [C] =>> NotGiven[C =:= Coordinate[Double]]](key: String): ValidatedNel[String, C] = 
            row.as[C](key).leftMap(_.getMessage).toValidatedNel
        private def getPair[C <: Coordinate[Double] : CellDecoder : [C] =>> NotGiven[C =:= Coordinate[Double]]](keyLo: String, keyHi: String): MaybeEndpoints[C] = 
            get[C](keyLo) -> get[C](keyHi)
        private def buildDecoderError(messages: NonEmptyList[String]): DecoderError = 
            DecoderError(s"${messages.size} error(s) decoding bounding box from row ($row): ${messages.mkString_("; ")}")

    private def buildBox(
        zBoundNels: MaybeEndpoints[ZCoordinate], 
        yBoundNels: MaybeEndpoints[YCoordinate], 
        xBoundNels: MaybeEndpoints[XCoordinate], 
    ): Either[NonEmptyList[String], BoundingBox] = 
        (zBoundNels._1, zBoundNels._2, yBoundNels._1, yBoundNels._2, xBoundNels._1, xBoundNels._2)
            .tupled
            .toEither
            .flatMap{ 
                (zLo, zHi, yLo, yHi, xLo, xHi) => 
                    val zIntervalNel = BoundingBox.Interval.fromTuple(zLo -> zHi).toValidatedNel
                    val yIntervalNel = BoundingBox.Interval.fromTuple(yLo -> yHi).toValidatedNel
                    val xIntervalNel = BoundingBox.Interval.fromTuple(xLo -> xHi).toValidatedNel
                    (xIntervalNel, yIntervalNel, zIntervalNel)
                        .mapN(BoundingBox.apply)
                        .toEither
            }
end SpatialCsvInstances
