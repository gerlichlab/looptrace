package at.ac.oeaw.imba.gerlich.looptrace.csv
package instances

import cats.*
import cats.data.*
import cats.syntax.all.*
import fs2.data.csv.*
import io.github.iltotore.iron.Constraint
import io.github.iltotore.iron.constraint.collection.MinLength
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2
import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2.syntax.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{
    ColumnNameLike, 
    IntraCellDelimiter,
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

import at.ac.oeaw.imba.gerlich.looptrace.collections.toNonEmptySet
import at.ac.oeaw.imba.gerlich.looptrace.RoiIndex
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.roi.*
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.{
    IndexedDetectedSpot,
    PostMergeRoi,
    RoiMergeBag,
}
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.RoiIndexColumnName

/** Typeclass instances related to CSV, for ROI-related data types */
trait RoiCsvInstances:
    private type Header = String
    
    private def roiIndexDelimiter = " "

    given (decRaw: CellDecoder[NonnegativeInt]) => CellDecoder[RoiIndex] = 
        decRaw.map(RoiIndex.apply)

    given cellEncoderForRoiIndex: CellEncoder[RoiIndex] = CellEncoder.fromSimpleShow[RoiIndex]

    given (CellDecoder[RoiIndex]) => CsvRowDecoder[RoiIndex, Header] = 
        getCsvRowDecoderForSingleton(ColumnNames.RoiIndexColumnName)

    given (CellEncoder[RoiIndex]) => CsvRowEncoder[RoiIndex, Header] = 
        getCsvRowEncoderForSingleton(ColumnNames.RoiIndexColumnName)

    given (decOne: CellDecoder[RoiIndex]) => CellDecoder[Set[RoiIndex]] = new:
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
    
    given (encOne: CellEncoder[RoiIndex]) => CellEncoder[Set[RoiIndex]] = new:
        override def apply(cell: Set[RoiIndex]): String = cell.map(encOne.apply).mkString(roiIndexDelimiter)

    given ( 
        CsvRowDecoder[DetectedSpot[Double], Header],
        CsvRowDecoder[BoundingBox, Header],
    ) => CsvRowDecoder[DetectedSpotRoi, Header] = 
        getCsvRowDecoderForProduct2(DetectedSpotRoi.apply)

    given ( 
        CsvRowEncoder[DetectedSpot[Double], Header], 
        CsvRowEncoder[BoundingBox, Header], 
    ) => CsvRowEncoder[DetectedSpotRoi, Header] = 
        getCsvRowEncoderForProduct2(_.spot, _.box)

    given ( 
        decIndex: CsvRowDecoder[RoiIndex, Header], 
        decContext: CsvRowDecoder[ImagingContext, Header], 
        decCentroid: CsvRowDecoder[Centroid[Double], Header], 
        decBox: CsvRowDecoder[BoundingBox, Header],
    ) => CsvRowDecoder[IndexedDetectedSpot, Header] = new:
        override def apply(row: RowF[Some, Header]): DecoderResult[IndexedDetectedSpot] = 
            val indexNel = decIndex(row)
                .leftMap(e => NonEmptyList.one(s"Problem decoding ROI index: $e"))
                .toValidated
            val contextNel = decContext(row)
                .leftMap(e => NonEmptyList.one(s"Problem decoding imaging context: $e"))
                .toValidated
            val centroidNel = decCentroid(row)
                .leftMap(e => NonEmptyList.one(s"Problem decoding spot/ROI centroid: $e"))
                .toValidated
            val boxNel = decBox(row)
                .leftMap(e => NonEmptyList.one(s"Problem decoding spot/ROI bounding box: $e"))
                .toValidated
            (indexNel, contextNel, centroidNel, boxNel)
                .mapN(IndexedDetectedSpot.apply)
                .leftMap{ errors => 
                    val context = "error(s) decoding indexed detected spot from CSV row"
                    DecoderError(s"${errors.length} $context ($row): ${errors.mkString_("; ")}") 
                }
                .toEither

    given ( 
        encIndex: CellEncoder[RoiIndex], 
        encContext: CsvRowEncoder[ImagingContext, Header], 
        encCentroid: CsvRowEncoder[Centroid[Double], Header], 
        encBox: CsvRowEncoder[BoundingBox, Header], 
    ) => CsvRowEncoder[IndexedDetectedSpot, Header] = new:
        override def apply(elem: IndexedDetectedSpot): RowF[Some, Header] = 
            val indexRow: NamedRow = RoiIndexColumnName.write(elem.index)
            val contextRow: NamedRow = encContext(elem.context)
            val centroidRow: NamedRow = encCentroid(elem.centroid)
            val boxRow: NamedRow = encBox(elem.box)
            indexRow |+| contextRow |+| centroidRow |+| boxRow

    given (
        CsvRowDecoder[IndexedDetectedSpot, Header], 
        CellDecoder[Set[RoiIndex]], 
    ) => CsvRowDecoder[MergerAssessedRoi, Header] = 
        given CsvRowDecoder[Set[RoiIndex], Header]:
            override def apply(row: RowF[Some, Header]): DecoderResult[Set[RoiIndex]] = 
                ColumnNames.MergeContributorsColumnNameForAssessedRecord.from(row)
                    .leftMap{ errorMessages => 
                        val context = s"problem(s) decoding merge indices from CSV row ($row)"
                        DecoderError(s"${errorMessages.length} $context: ${errorMessages.mkString_("; ")}")
                    }
                    .toEither
        getCsvRowDecoderForProduct2{ (spot: IndexedDetectedSpot, mergeInputs: Set[RoiIndex]) => 
            MergerAssessedRoi.build(spot, mergeInputs).fold(errMsg => throw new DecoderError(errMsg), identity)
        }

    given ( 
        encSpot: CsvRowEncoder[IndexedDetectedSpot, Header], 
        encRoiIndices: CellEncoder[Set[RoiIndex]],
    ) => CsvRowEncoder[MergerAssessedRoi, Header] = 
        given CsvRowEncoder[Set[RoiIndex], Header]:
            override def apply(elem: Set[RoiIndex]): RowF[Some, Header] = 
                ColumnNames.MergeContributorsColumnNameForAssessedRecord.write(elem)
        getCsvRowEncoderForProduct2(_.spot, _.mergeNeighbors)

    given ( 
        decIndex: CellDecoder[RoiIndex],
        decRoi: CsvRowDecoder[DetectedSpotRoi, Header], 
    ) => CsvRowDecoder[MergeContributorRoi, Header] = new:
        override def apply(row: RowF[Some, Header]): DecoderResult[MergeContributorRoi] = 
            val indexNel: ValidatedNel[String, RoiIndex] = ColumnNames.RoiIndexColumnName.from(row)
            val roiNel: ValidatedNel[String, DetectedSpotRoi] = parseDetectedSpotRoi(row)
            val mergeOutputNel: ValidatedNel[String, RoiIndex] = ColumnNames.MergeOutputColumnName.from(row)
            (indexNel, roiNel, mergeOutputNel)
                .tupled
                .toEither
                .flatMap{ (index, roi, mergeOutput) => 
                    MergeContributorRoi.apply(index, roi, mergeOutput).leftMap(NonEmptyList.one)
                }
                .leftMap{ messages => 
                    DecoderError(s"${messages.length} error(s) decoding merge contributor ROI: ${messages.mkString_("; ")}") 
                }

    given (
        encIndex: CellEncoder[RoiIndex], 
        encContext: CsvRowEncoder[ImagingContext, Header], 
        encCentroid: CsvRowEncoder[Centroid[Double], Header],
        encBox: CsvRowEncoder[BoundingBox, Header],
    ) => CsvRowEncoder[MergeContributorRoi, Header] = new:
        override def apply(elem: MergeContributorRoi): RowF[Some, Header] = 
            val idRow: NamedRow = ColumnNames.RoiIndexColumnName.write(elem.index)
            val contextRow: NamedRow = encContext(elem.context)
            val centroidRow: NamedRow = encCentroid(elem.centroid)
            val boxRow: NamedRow = encBox(elem.box)
            val mergeOutputRow: NamedRow = ColumnNames.MergeOutputColumnName.write(elem.mergeOutput)
            idRow |+| contextRow |+| centroidRow |+| boxRow |+| mergeOutputRow

    def parseFromRow[A](messagePrefix: String)(using dec: CsvRowDecoder[A, Header]): RowF[Some, Header] => ValidatedNel[String, A] = 
        row => dec(row)
            .leftMap{ e => s"$messagePrefix: ${e.getMessage}" }
            .toValidatedNel
        
    def parseDetectedSpotRoi(using CsvRowDecoder[DetectedSpotRoi, Header]): RowF[Some, Header] => ValidatedNel[String, DetectedSpotRoi] = 
        parseFromRow("Error decoding ROI")

    given ( 
        decIndex: CellDecoder[RoiIndex], 
        decContext: CsvRowDecoder[ImagingContext, Header], 
        decCentroid: CsvRowDecoder[Centroid[Double], Header], 
        decBox: CsvRowDecoder[BoundingBox, Header], 
    ) => CsvRowDecoder[MergedRoiRecord, Header] = 
        import at.ac.oeaw.imba.gerlich.looptrace.collections.toNonEmptySet
        given Ordering[RoiIndex] = summon[Order[RoiIndex]].toOrdering
        given CellDecoder[NonEmptySet[RoiIndex]] = new: // to build the CsvRowDecoder
            override def apply(cell: String): DecoderResult[NonEmptySet[RoiIndex]] = 
                summon[CellDecoder[Set[RoiIndex]]](cell)
                    .flatMap(_.toNonEmptySet.toRight{ DecoderError("Empty collection of merge contributors") })
        new:
            override def apply(row: RowF[Some, Header]): DecoderResult[MergedRoiRecord] = 
                val indexNel = ColumnNames.RoiIndexColumnName.from(row)
                val contextNel = decContext(row)
                    .leftMap(e => s"Problem parsing imaging context: ${e.getMessage}")
                    .toValidatedNel
                val centroidNel = decCentroid(row)
                    .leftMap(e => s"Problem parsing centroid: ${e.getMessage}")
                    .toValidatedNel
                val boxNel = decBox(row)
                    .leftMap(e => s"Problem parsing bounding box: ${e.getMessage}")
                    .toValidatedNel
                val contributorsNel = 
                    ColumnNames.MergeContributorsColumnNameForMergedRecord.from(row)
                (indexNel, contextNel, centroidNel, boxNel, contributorsNel)
                    .mapN(MergedRoiRecord.apply)
                    .toEither
                    .leftMap{ messages => DecoderError(s"${messages.length} error(s) decoding merged ROI record: ${messages.mkString_("; ")}") }

    private inline given [C[*], X] => ( 
        decCX: CellDecoder[C[X]], 
        inline constraint: Constraint[C[X], MinLength[2]],
    ) => CellDecoder[AtLeast2[C, X]] = 
        decCX.emap{ AtLeast2.either(_).leftMap(msg => DecoderError(msg)) }

    given ( 
        encIndex: CellEncoder[RoiIndex], 
        encContext: CsvRowEncoder[ImagingContext, Header],
        encCentroid: CellEncoder[Double],
        encRoiBag: CellEncoder[Set[RoiIndex]],
    ) => CsvRowEncoder[MergedRoiRecord, Header] = 
        new:
            override def apply(elem: MergedRoiRecord): RowF[Some, Header] = 
                val idxRow: NamedRow = ColumnNames.RoiIndexColumnName.write(elem.index)
                val ctxRow: NamedRow = encContext(elem.context)
                val centroidRow: NamedRow = summon[CsvRowEncoder[Centroid[Double], Header]](elem.centroid)
                val contribsRow: NamedRow = 
                    given IntraCellDelimiter[RoiIndex] = IntraCellDelimiter(roiIndexDelimiter)
                    ColumnNames.MergeContributorsColumnNameForMergedRecord.write(elem.contributors)
                idxRow |+| ctxRow |+| centroidRow |+| contribsRow

    given (
        encIndex: CellEncoder[RoiIndex], 
        encContext: CsvRowEncoder[ImagingContext, Header],
        encCentroid: CellEncoder[Double],
        encBox: CsvRowEncoder[BoundingBox, String],
        encRoiBag: CellEncoder[Set[RoiIndex]],
    ) => CsvRowEncoder[UnidentifiableRoi, Header] = 
        val encTooClose: CsvRowEncoder[NonEmptySet[RoiIndex], Header] = 
            getCsvRowEncoderForSingleton(ColumnNames.TooCloseRoisColumnName)(using encRoiBag.contramap(_.toSortedSet))
        new:
            override def apply(elem: UnidentifiableRoi): RowF[Some, Header] = 
                val idxRow: NamedRow = ColumnNames.RoiIndexColumnName.write(elem.index)
                val ctxRow: NamedRow = encContext(elem.context)
                val centroidRow: NamedRow = summon[CsvRowEncoder[Centroid[Double], Header]](elem.centroid)
                val boxRow: NamedRow = encBox(elem.box)
                val tooCloseRow: NamedRow = encTooClose(elem.tooClose)
                idxRow |+| ctxRow |+| centroidRow |+| boxRow |+| tooCloseRow

    
    given ( 
        decIndex: CellDecoder[RoiIndex],
        decSpot: CsvRowDecoder[IndexedDetectedSpot, String], 
    ) => CsvRowDecoder[PostMergeRoi, String] = 
        given Ordering[RoiIndex] = summon[Order[RoiIndex]].toOrdering
        new:
            override def apply(row: RowF[Some, String]): DecoderResult[PostMergeRoi] = 
                val spotNel = parseFromRow[IndexedDetectedSpot]("Error(s) decoding ROI")(row)
                val contributorsNel = ColumnNames.MergeContributorsColumnNameForAssessedRecord.from(row)
                (spotNel, contributorsNel)
                    .tupled
                    .toEither
                    .flatMap{ (spot, contributors) => 
                        if contributors.size === 0 then spot.asRight
                        else AtLeast2.either(contributors).bimap(
                            NonEmptyList.one, 
                            MergedRoiRecord(spot, _)
                        )
                    }
                    .leftMap{ messages => 
                        DecoderError(s"${messages.length} error(s) decoding ROI: ${messages.mkString_("; ")}") 
                    }

    given ( 
        encIdx: CellEncoder[RoiIndex],
        encContext: CsvRowEncoder[ImagingContext, String], 
        encCentroid: CsvRowEncoder[Centroid[Double], String], 
        encBox: CsvRowEncoder[BoundingBox, String],
    ) => CsvRowEncoder[PostMergeRoi, String] = 
        import AdmitsRoiIndex.*
        import IndexedDetectedSpot.given
        import PostMergeRoi.*
        import PostMergeRoi.given
        new:
            override def apply(elem: PostMergeRoi): RowF[Some, String] = 
                val idxRow: NamedRow = ColumnNames.RoiIndexColumnName.write(elem.roiIndex)
                val ctxRow: NamedRow = encContext(elem.context)
                val (centroid, box) = PostMergeRoi.getCenterAndBox(elem)
                val contribs: Set[RoiIndex] = elem match {
                    case _: IndexedDetectedSpot => Set()
                    case merged: MergedRoiRecord => merged.contributors.toSet
                }
                val mergeRoisRow: NamedRow = ColumnNames.MergeContributorsColumnNameForAssessedRecord.write(contribs)
                idxRow |+| ctxRow |+| encCentroid(centroid) |+| encBox(box) |+| mergeRoisRow
end RoiCsvInstances
