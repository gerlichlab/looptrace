package at.ac.oeaw.imba.gerlich.looptrace
package csv
package instances

import scala.util.Try
import cats.syntax.all.*
import fs2.data.csv.{ CellDecoder, CsvRowDecoder, DecoderError, DecoderResult, RowF }

import at.ac.oeaw.imba.gerlich.gerlib.geometry.{
    XCoordinate as XC, 
    YCoordinate as YC, 
    ZCoordinate as ZC, 
}
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
    /** Attempt to parse the given string as a double. */
    private def readDouble: String => Either[String, Double] = s => 
        Try{ s.toDouble }
            .toEither
            .leftMap(Function.const{ s"Cannot read as double: $s" })
    
    /** Create a [[fs2.data.csv.CellDecoder]] from this trait' double parser. */
    private given CellDecoder[Double] = liftToCellDecoder(readDouble)

    given CsvRowDecoder[BoundingBox, String] with
        /** Parse each interval endpoint, then make assemble the intervals. */
        override def apply(row: RowF[Some, String]): DecoderResult[BoundingBox] = 
            val zLoNel = row.as[ZCoordinate]("zMin").toValidatedNel
            val zHiNel = row.as[ZCoordinate]("zMax").toValidatedNel
            val yLoNel = row.as[YCoordinate]("yMin").toValidatedNel
            val yHiNel = row.as[YCoordinate]("yMax").toValidatedNel
            val xLoNel = row.as[XCoordinate]("xMin").toValidatedNel
            val xHiNel = row.as[XCoordinate]("xMax").toValidatedNel
            (zLoNel, zHiNel, yLoNel, yHiNel, xLoNel, xHiNel)
                .tupled
                .toEither
                .leftMap(_.map(_.getMessage)) // convert errors to messages.
                .flatMap{ 
                    (zLo, zHi, yLo, yHi, xLo, xHi) => 
                        val zIntervalNel = BoundingBox.Interval.fromTuple(zLo -> zHi).toValidatedNel
                        val yIntervalNel = BoundingBox.Interval.fromTuple(yLo -> yHi).toValidatedNel
                        val xIntervalNel = BoundingBox.Interval.fromTuple(xLo -> xHi).toValidatedNel
                        (xIntervalNel, yIntervalNel, zIntervalNel)
                            .mapN(BoundingBox.apply)
                            .toEither
                }
                .leftMap{ messages => 
                    val msg = s"${messages.size} error(s) decoding bounding box from row ($row): ${messages.mkString_("; ")}"
                    DecoderError(msg)
                }

end SpatialCsvInstances
