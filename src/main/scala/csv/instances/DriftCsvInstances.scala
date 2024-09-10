package at.ac.oeaw.imba.gerlich.looptrace.csv
package instances

import scala.util.NotGiven
import cats.syntax.all.*
import fs2.data.csv.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanAxis
import at.ac.oeaw.imba.gerlich.looptrace.drift.*

/** CSV-related typeclass instances related to microscope drift */
trait DriftCsvInstances:
    private trait Buildable

    given cellDecoderForCoarseComponent[A <: EuclideanAxis: [A] =>> NotGiven[A =:= EuclideanAxis]](
        using decInt: CellDecoder[Int]
    ): CellDecoder[CoarseDriftComponent[A]] = 
        decInt.map(DriftComponent.coarse)

    given cellDecoderForFineComponent[A <: EuclideanAxis: [A] =>> NotGiven[A =:= EuclideanAxis]](
        using decInt: CellDecoder[Int]
    ): CellDecoder[FineDriftComponent[A]] = 
        decInt.map(DriftComponent.fine)
    
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
end DriftCsvInstances
