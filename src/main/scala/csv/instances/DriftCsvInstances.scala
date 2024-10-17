package at.ac.oeaw.imba.gerlich.looptrace.csv
package instances

import scala.util.NotGiven
import cats.syntax.all.*
import fs2.data.csv.*

import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanAxis
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.roi.parseFromRow
import at.ac.oeaw.imba.gerlich.looptrace.drift.*

/** CSV-related typeclass instances related to microscope drift */
trait DriftCsvInstances:
    private trait Buildable

    given cellDecoderForCoarseComponent[A <: EuclideanAxis: [A] =>> NotGiven[A =:= EuclideanAxis]](
        using decInt: CellDecoder[Int]
    ): CellDecoder[CoarseDriftComponent[A]] = 
        decInt.map(DriftComponent.coarse)

    given cellDecoderForFineComponent[A <: EuclideanAxis: [A] =>> NotGiven[A =:= EuclideanAxis]](
        using decDecimal: CellDecoder[Double]
    ): CellDecoder[FineDriftComponent[A]] = 
        decDecimal.map(DriftComponent.fine)
    
    given csvRowDecoderForCoarseDrift(using CellDecoder[Int]): CsvRowDecoder[CoarseDrift, String] with
        override def apply(row: RowF[Some, String]): DecoderResult[CoarseDrift] = 
            val xNel = ColumnNames.CoarseDriftColumnNameX.from(row)
            val yNel = ColumnNames.CoarseDriftColumnNameY.from(row)
            val zNel = ColumnNames.CoarseDriftColumnNameZ.from(row)
            (xNel, yNel, zNel)
                .mapN{ (x, y, z) => CoarseDrift(z, y, x) }
                .toEither
                .leftMap{ messages => DecoderError(
                    s"Tried to parse coarse drift from CSV row, but got ${messages.length} error(s): ${messages.mkString_("; ")}"
                ) }

    given csvRowDecoderForFineDrift(using CellDecoder[Double]): CsvRowDecoder[FineDrift, String] with
        override def apply(row: RowF[Some, String]): DecoderResult[FineDrift] = 
            val xNel = ColumnNames.FineDriftColumnNameX.from(row)
            val yNel = ColumnNames.FineDriftColumnNameY.from(row)
            val zNel = ColumnNames.FineDriftColumnNameZ.from(row)
            (xNel, yNel, zNel)
                .mapN{ (x, y, z) => FineDrift(z, y, x) }
                .toEither
                .leftMap{ messages => DecoderError(
                    s"Tried to parse coarse drift from CSV row, but got ${messages.length} error(s): ${messages.mkString_("; ")}"
                ) }

    given decoderForDriftRecord(using 
        CsvRowDecoder[FieldOfViewLike, String], 
        CsvRowDecoder[ImagingTimepoint, String],
        CsvRowDecoder[CoarseDrift, String], 
        CsvRowDecoder[FineDrift, String],
    ): CsvRowDecoder[DriftRecord, String] with
        override def apply(row: RowF[Some, String]): DecoderResult[DriftRecord] = 
            val fovNel = parseFromRow[FieldOfViewLike]("Error(s) reading field of view from CSV row")(row)
            val timeNel = parseFromRow[ImagingTimepoint]("Error(s) reading timepoint from CSV row")(row)
            val coarseDriftNel = parseFromRow[CoarseDrift]("Error(s) reading coarse drift from CSV row")(row)
            val fineDriftNel = parseFromRow[FineDrift]("Error(s) reading fine drift from CSV row")(row)
            (fovNel, timeNel, coarseDriftNel, fineDriftNel).mapN{ (fov, time, coarseDrift, fineDrift) =>  
                DriftRecord(fov, time, coarseDrift, fineDrift)
            }
            .leftMap{ messages => 
                DecoderError(s"Failed to parse drift record from CSV row. Messages: ${messages.mkString_("; ")}")
            }
            .toEither
end DriftCsvInstances
