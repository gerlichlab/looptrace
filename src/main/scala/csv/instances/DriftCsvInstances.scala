package at.ac.oeaw.imba.gerlich.looptrace.csv
package instances

import scala.util.NotGiven
import cats.syntax.all.*
import mouse.boolean.*
import fs2.data.csv.*

import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanAxis
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.roi.parseFromRow
import at.ac.oeaw.imba.gerlich.looptrace.drift.*

/** CSV-related typeclass instances related to microscope drift */
trait DriftCsvInstances:
  private trait Buildable

  given [A <: EuclideanAxis] => (
      nonGeneric: NotGiven[A =:= EuclideanAxis],
      decInt: CellDecoder[Int],
      decDecimal: CellDecoder[Double]
  ) => CellDecoder[CoarseDriftComponent[A]] =
    val decimalIsInt = (x: Double) => x.floor.toInt === x.ceil.toInt
    val decimalIntDecoder: CellDecoder[Int] = new:
      override def apply(cell: String): DecoderResult[Int] =
        decDecimal(cell) >>= { (x: Double) =>
          decimalIsInt(x).either(
            DecoderError(s"Unable to decode '$cell' as integer"),
            x.toInt
          )
        }
    decInt.orElse(decimalIntDecoder).map(DriftComponent.coarse)

  given [A <: EuclideanAxis] => (
      nonGeneric: NotGiven[A =:= EuclideanAxis],
      decDecimal: CellDecoder[Double]
  ) => CellDecoder[FineDriftComponent[A]] =
    decDecimal.map(DriftComponent.fine)

  given (CellDecoder[Int]) => CsvRowDecoder[CoarseDrift, String]:
    override def apply(row: RowF[Some, String]): DecoderResult[CoarseDrift] =
      val xNel = ColumnNames.CoarseDriftColumnNameX.from(row)
      val yNel = ColumnNames.CoarseDriftColumnNameY.from(row)
      val zNel = ColumnNames.CoarseDriftColumnNameZ.from(row)
      (xNel, yNel, zNel)
        .mapN { (x, y, z) => CoarseDrift(z, y, x) }
        .toEither
        .leftMap { messages =>
          DecoderError(
            s"Tried to parse coarse drift from CSV row, but got ${messages.length} error(s): ${messages.mkString_("; ")}"
          )
        }

  given (CellDecoder[Double]) => CsvRowDecoder[FineDrift, String]:
    override def apply(row: RowF[Some, String]): DecoderResult[FineDrift] =
      val xNel = ColumnNames.FineDriftColumnNameX.from(row)
      val yNel = ColumnNames.FineDriftColumnNameY.from(row)
      val zNel = ColumnNames.FineDriftColumnNameZ.from(row)
      (xNel, yNel, zNel)
        .mapN { (x, y, z) => FineDrift(z, y, x) }
        .toEither
        .leftMap { messages =>
          DecoderError(
            s"Tried to parse coarse drift from CSV row, but got ${messages.length} error(s): ${messages.mkString_("; ")}"
          )
        }

  given (
      CsvRowDecoder[FieldOfViewLike, String],
      CsvRowDecoder[ImagingTimepoint, String],
      CsvRowDecoder[CoarseDrift, String],
      CsvRowDecoder[FineDrift, String]
  ) => CsvRowDecoder[DriftRecord, String]:
    override def apply(row: RowF[Some, String]): DecoderResult[DriftRecord] =
      val fovNel = parseFromRow[FieldOfViewLike](
        "Error(s) reading field of view from CSV row"
      )(row)
      val timeNel = parseFromRow[ImagingTimepoint](
        "Error(s) reading timepoint from CSV row"
      )(row)
      val coarseDriftNel = parseFromRow[CoarseDrift](
        "Error(s) reading coarse drift from CSV row"
      )(row)
      val fineDriftNel =
        parseFromRow[FineDrift]("Error(s) reading fine drift from CSV row")(row)
      (fovNel, timeNel, coarseDriftNel, fineDriftNel)
        .mapN { (fov, time, coarseDrift, fineDrift) =>
          DriftRecord(fov, time, coarseDrift, fineDrift)
        }
        .leftMap { messages =>
          DecoderError(
            s"Failed to parse drift record from CSV row. Messages: ${messages.mkString_("; ")}"
          )
        }
        .toEither
end DriftCsvInstances
