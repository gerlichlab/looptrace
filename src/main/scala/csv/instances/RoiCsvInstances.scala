package at.ac.oeaw.imba.gerlich.looptrace.csv
package instances

import cats.*
import cats.data.*
import cats.syntax.all.*
import fs2.data.csv.*
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{
    ColumnNameLike, 
    NamedRow,
    getCsvRowDecoderForProduct2, 
    getCsvRowDecoderForSingleton,
    getCsvRowEncoderForProduct2,
    getCsvRowEncoderForSingleton,
    fromSimpleShow
}
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.roi.DetectedSpot
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.RoiIndex
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.roi.*
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox
import at.ac.oeaw.imba.gerlich.gerlib.imaging.FieldOfViewLike
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingChannel

/** Typeclass instances related to CSV, for ROI-related data types */
trait RoiCsvInstances:
    private type Header = String
    
    private def roiIndexDelimiter = ";"

    given cellDecoderForRoiIndex(using decRaw: CellDecoder[NonnegativeInt]): CellDecoder[RoiIndex] = 
        decRaw.map(RoiIndex.apply)

    given cellEncoderForRoiIndex: CellEncoder[RoiIndex] = CellEncoder.fromSimpleShow[RoiIndex]

    given csvRowDecoderForRoiIndex(using CellDecoder[RoiIndex]): CsvRowDecoder[RoiIndex, Header] = 
        getCsvRowDecoderForSingleton(ColumnNames.RoiIndexColumnName)

    given csvRowEncoderForRoiIndex(using CellEncoder[RoiIndex]): CsvRowEncoder[RoiIndex, Header] = 
        getCsvRowEncoderForSingleton(ColumnNames.RoiIndexColumnName)

    given cellDecoderForRoiBag(using decOne: CellDecoder[RoiIndex]): CellDecoder[Set[RoiIndex]] = new:
        override def apply(cell: String): DecoderResult[Set[RoiIndex]] = 
            if cell === "" 
            then Set.empty.asRight 
            else cell.split(roiIndexDelimiter)
                .toList
                .traverse(decOne.apply)
                .flatMap{ allIndices => 
                    val repeats = allIndices.groupBy(identity).view.mapValues(_.length).filter(_._2 > 1).toMap
                    repeats.isEmpty.either(
                        DecoderError(s"Repeat counts by ROI index: $repeats"), 
                        allIndices.toSet
                    )
                }
    
    given cellEncoderForRoiBag(using encOne: CellEncoder[RoiIndex]): CellEncoder[Set[RoiIndex]] = new:
        override def apply(cell: Set[RoiIndex]): String = cell.map(encOne.apply).mkString(roiIndexDelimiter)

    summon[CsvRowDecoder[DetectedSpot[Double], Header]]
    summon[CsvRowDecoder[BoundingBox, Header]]

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

    given csvRowDecoderForMergerAssessedRoi(using 
        decIndex: CsvRowDecoder[RoiIndex, Header],
        decRoi: CsvRowDecoder[DetectedSpotRoi, Header], 
        decRoiIndices: CellDecoder[Set[RoiIndex]], 
    ): CsvRowDecoder[MergerAssessedRoi, Header] = new:
        override def apply(row: RowF[Some, Header]): DecoderResult[MergerAssessedRoi] = 
            val indexNel = decIndex(row)
                .toValidatedNel
                .leftMap(es => NonEmptyList.one(s"${es.size} problem(s) with main ROI index: $es"))
            val roiNel = decRoi(row)
                .toValidatedNel.leftMap(_.map(_.getMessage))
                .leftMap(es => NonEmptyList.one(s"${es.size} problem(s) with detected ROI: $es"))
            val mergeNel = ColumnNames.MergeRoisColumnName
                .from(row)
                .leftMap(es => NonEmptyList.one(s"${es.size} problem(s) with merge ROIs: $es"))
            (indexNel, roiNel, mergeNel)
                .tupled
                .toEither
                .flatMap{ (index, roi, merge) => 
                    MergerAssessedRoi.build(index, roi, merge = merge).leftMap(NonEmptyList.one)
                }
                .leftMap{ messages => 
                    val context = "error(s) decoding nucleus-labeled, proximity-assessed ROI from CSV row"
                    DecoderError(s"${messages.length} $context ($row): ${messages.mkString_("; ")}")
                }

    given csvRowEncoderForMergerAssessedRoi(using 
        encRoi: CsvRowEncoder[DetectedSpotRoi, Header], 
        encIndex: CsvRowEncoder[RoiIndex, Header],
        encRoiIndices: CellEncoder[Set[RoiIndex]],
    ): CsvRowEncoder[MergerAssessedRoi, Header] = new:
        override def apply(elem: MergerAssessedRoi): RowF[Some, Header] = 
            val iRow: NamedRow = encIndex(elem.index)
            val init: NamedRow = encRoi(elem.roi)
            val extra: NamedRow = NamedRow(
                Some(NonEmptyList.one(ColumnNames.MergeRoisColumnName.value)),
                NonEmptyList.one(encRoiIndices(elem.mergeNeighbors)),
            )
            iRow |+| init |+| extra

    given csvRowDecoderForMergeContributorRoi(using 
        decIndex: CsvRowDecoder[RoiIndex, Header], 
        decRoi: CsvRowDecoder[DetectedSpotRoi, Header], 
    ): CsvRowDecoder[MergeContributorRoi, Header] = new:
        override def apply(row: RowF[Some, Header]): DecoderResult[MergeContributorRoi] = 
            val indexNel: ValidatedNel[String, RoiIndex] = ColumnNames.RoiIndexColumnName.from(row)
            val roiNel: ValidatedNel[String, DetectedSpotRoi] = parseDetectedSpotRoi(row)
            val mergeIndexNel: ValidatedNel[String, RoiIndex] = ColumnNames.MergedIndexColumnName.from(row)
            (indexNel, roiNel, mergeIndexNel)
                .tupled
                .toEither
                .flatMap{ (index, roi, mergeIndex) => 
                    MergeContributorRoi.apply(index, roi, mergeIndex).leftMap(NonEmptyList.one)
                }
                .leftMap{ messages => 
                    DecoderError(s"${messages.length} error(s) decoding merge contributor ROI: ${messages.mkString_("; ")}") 
                }

    def parseFromRow[A](messagePrefix: String)(using dec: CsvRowDecoder[A, Header]): RowF[Some, Header] => ValidatedNel[String, A] = 
        row => dec(row)
            .leftMap{ e => s"$messagePrefix: ${e.getMessage}" }
            .toValidatedNel
        
    def parseDetectedSpotRoi(using CsvRowDecoder[DetectedSpotRoi, Header]): RowF[Some, Header] => ValidatedNel[String, DetectedSpotRoi] = 
        parseFromRow("Error decoding ROI")

    given csvRowDecoderForMergedRoiRecord(using 
        CellDecoder[RoiIndex], 
        CsvRowDecoder[DetectedSpotRoi, Header], 
    ): CsvRowDecoder[MergedRoiRecord, Header] = 
        import at.ac.oeaw.imba.gerlich.looptrace.collections.toNonEmptySet
        given Ordering[RoiIndex] = summon[Order[RoiIndex]].toOrdering
        given CellDecoder[NonEmptySet[RoiIndex]] = new: // to build the CsvRowDecoder
            override def apply(cell: String): DecoderResult[NonEmptySet[RoiIndex]] = 
                summon[CellDecoder[Set[RoiIndex]]](cell)
                    .flatMap(_.toNonEmptySet.toRight{ DecoderError("Empty collection of merge contributors") })
        new:
            override def apply(row: RowF[Some, Header]): DecoderResult[MergedRoiRecord] = 
                val indexNel = ColumnNames.RoiIndexColumnName.from(row)
                val roiNel = parseDetectedSpotRoi(row)
                val contributorsNel = ColumnNames.MergeContributorsColumnName.from(row)
                (indexNel, roiNel, contributorsNel)
                    .mapN(MergedRoiRecord.apply)
                    .toEither
                    .leftMap{ messages => DecoderError(s"${messages.length} error(s) decoding merged ROI record: ${messages.mkString_("; ")}") }

    given csvRowEncoderForMergedRoiRecord(using 
        encIndex: CellEncoder[RoiIndex], 
        encContext: CsvRowEncoder[ImagingContext, Header],
        encCentroid: CellEncoder[Double],
        encRoiBag: CellEncoder[Set[RoiIndex]],
    ): CsvRowEncoder[MergedRoiRecord, Header] = 
        val encContribs: CsvRowEncoder[NonEmptySet[RoiIndex], Header] = {
            given CellEncoder[NonEmptySet[RoiIndex]] = encRoiBag.contramap((_: NonEmptySet[RoiIndex]).toSortedSet)
            getCsvRowEncoderForSingleton(ColumnNames.MergeContributorsColumnName)
        }
        new:
            override def apply(elem: MergedRoiRecord): RowF[Some, Header] = 
                val idxRow: NamedRow = ColumnNames.RoiIndexColumnName.write(elem.index)
                val ctxRow: NamedRow = encContext(elem.context)
                val centroidRow: NamedRow = summon[CsvRowEncoder[Centroid[Double], Header]](elem.centroid)
                val contribsRow: NamedRow = encContribs(elem.contributors)
                idxRow |+| ctxRow |+| centroidRow |+| contribsRow
end RoiCsvInstances
