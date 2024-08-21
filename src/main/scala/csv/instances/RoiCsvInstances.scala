package at.ac.oeaw.imba.gerlich.looptrace.csv
package instances

import cats.data.*
import cats.syntax.all.*
import fs2.data.csv.*
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{
    getCsvRowDecoderForProduct2, 
    getCsvRowDecoderForSingleton,
    getCsvRowEncoderForProduct2,
}
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.roi.DetectedSpot
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.{
    DetectedSpotRoi, 
    NucleusLabelAttemptedRoi,
    NucleusLabeledProximityAssessedRoi, 
    RoiIndex, 
}
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.TooCloseRoisColumnName
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt

/** Typeclass instances related to CSV, for ROI-related data types */
trait RoiCsvInstances:
    private type Header = String
    
    given cellDecoderForRoiIndex(using decRaw: CellDecoder[NonnegativeInt]): CellDecoder[RoiIndex] = 
        decRaw.map(RoiIndex.apply)

    given csvRowDecoderForRoiIndex(using CellDecoder[RoiIndex]): CsvRowDecoder[RoiIndex, Header] = 
        getCsvRowDecoderForSingleton(ColumnNames.RoiIndexColumnName)

    given cellDecoderForRoiBag(using decOne: CellDecoder[RoiIndex]): CellDecoder[Set[RoiIndex]] = new:
        override def apply(cell: String): DecoderResult[Set[RoiIndex]] = 
            cell.split(";")
                .toList
                .traverse(decOne.apply)
                .flatMap{ allIndices => 
                    val repeats = allIndices.groupBy(identity).view.mapValues(_.length).filter(_._2 > 1).toMap
                    repeats.isEmpty.either(
                        DecoderError(s"Repeat counts by ROI index: $repeats"), 
                        repeats.keySet
                    )
                }
        

    given csvRowDecoderForDetectedSpotRoi(using 
        CsvRowDecoder[DetectedSpot[Double], Header],
        CsvRowDecoder[BoundingBox, Header],
    ): CsvRowDecoder[DetectedSpotRoi, Header] = 
        getCsvRowDecoderForProduct2(DetectedSpotRoi.apply)

    given csvRowEncoderForDetectedSpotRoi(using 
        CsvRowEncoder[DetectedSpot[Double], Header], 
        CsvRowEncoder[BoundingBox, Header], 
    ): CsvRowEncoder[DetectedSpotRoi, Header] = 
        getCsvRowEncoderForProduct2(_.spot, _.box)

    given csvRowDecoderForNucleusLabelAttemptedRoi(using 
        CsvRowDecoder[DetectedSpotRoi, Header],
        CsvRowDecoder[NuclearDesignation, Header]
    ): CsvRowDecoder[NucleusLabelAttemptedRoi, Header] = 
        getCsvRowDecoderForProduct2(NucleusLabelAttemptedRoi.apply)

    given csvRowEncoderForNucleusLabelAttemptedRoi(using 
        CsvRowEncoder[DetectedSpotRoi, Header], 
        CsvRowEncoder[NuclearDesignation, Header]
    ): CsvRowEncoder[NucleusLabelAttemptedRoi, Header] = 
        getCsvRowEncoderForProduct2(
            _.toDetectedSpotRoi,
            _.nucleus
        )

    given csvRowDecoderForNucleusLabeledProximityAssessedRoi(using 
        decIndex: CsvRowDecoder[RoiIndex, Header],
        decRoi: CsvRowDecoder[NucleusLabelAttemptedRoi, Header], 
        decRoiIndices: CellDecoder[Set[RoiIndex]], 
    ): CsvRowDecoder[NucleusLabeledProximityAssessedRoi, Header] = new:
        override def apply(row: RowF[Some, Header]): DecoderResult[NucleusLabeledProximityAssessedRoi] = 
            val indexNel = decIndex(row).toValidatedNel.leftMap(_.map(_.getMessage))
            val roiNel = decRoi(row).toValidatedNel.leftMap(_.map(_.getMessage))
            val tooCloseNel = ColumnNames.TooCloseRoisColumnName.from(row)
            val mergeNel = ColumnNames.MergeRoisColumnName.from(row)
            (indexNel, roiNel, tooCloseNel, mergeNel)
                .tupled
                .toEither
                .flatMap{ (index, roi, tooClose, merge) => 
                    NucleusLabeledProximityAssessedRoi.build(
                        index, 
                        roi.toDetectedSpotRoi, 
                        roi.nucleus,
                        tooClose = tooClose, 
                        merge = merge,
                    )
                }
                .leftMap{ messages => 
                    val context = "error(s) decoding nucleus-labeled, proximity-assessed ROI from CSV row"
                    DecoderError(s"${messages.length} $context ($row): ${messages.mkString_("; ")}")
                }

    given csvRowEncoderForNucleusLabeledProximityAssessedRoi(using 
        encRoi: CsvRowEncoder[NucleusLabelAttemptedRoi, Header], 
        encIndex: CsvRowEncoder[RoiIndex, Header],
        encRoiIndices: CellEncoder[Set[RoiIndex]],
    ): CsvRowEncoder[NucleusLabeledProximityAssessedRoi, Header] = new:
        override def apply(elem: NucleusLabeledProximityAssessedRoi): RowF[Some, Header] = 
            val init: RowF[Some, Header] = encRoi(elem.dropNeighbors)
            val extra = NonEmptyList.of(
                ColumnNames.TooCloseRoisColumnName -> encRoiIndices(elem.tooCloseNeighbors), 
                ColumnNames.MergeRoisColumnName -> encRoiIndices(elem.mergeNeighbors), 
            )
            val (extraCols, extraValues) = extra.unzip
            val iRow = encIndex(elem.index)
            RowF(
                values = iRow.values ::: init.values ::: extraValues,
                headers = Some(iRow.headers.extractValue ::: init.headers.extractValue ::: extraCols.map(_.value)), 
            )
end RoiCsvInstances
