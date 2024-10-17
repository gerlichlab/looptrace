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

import at.ac.oeaw.imba.gerlich.looptrace.collections.toNonEmptySet
import at.ac.oeaw.imba.gerlich.looptrace.RoiIndex
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.roi.*
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.{
    IndexedDetectedSpot,
    PostMergeRoi,
}
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.RoiIndexColumnName

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

    given csvRowDecoderForIndexedDetectedSpot(using 
        decIndex: CsvRowDecoder[RoiIndex, Header], 
        decContext: CsvRowDecoder[ImagingContext, Header], 
        decCentroid: CsvRowDecoder[Centroid[Double], Header], 
        decBox: CsvRowDecoder[BoundingBox, Header],
    ): CsvRowDecoder[IndexedDetectedSpot, Header] = new:
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

    given csvRowEncoderForIndexedDetectedSpot(using 
        encIndex: CellEncoder[RoiIndex], 
        encContext: CsvRowEncoder[ImagingContext, Header], 
        encCentroid: CsvRowEncoder[Centroid[Double], Header], 
        encBox: CsvRowEncoder[BoundingBox, Header], 
    ): CsvRowEncoder[IndexedDetectedSpot, Header] = new:
        override def apply(elem: IndexedDetectedSpot): RowF[Some, Header] = 
            val indexRow: NamedRow = RoiIndexColumnName.write(elem.index)
            val contextRow: NamedRow = encContext(elem.context)
            val centroidRow: NamedRow = encCentroid(elem.centroid)
            val boxRow: NamedRow = encBox(elem.box)
            indexRow |+| contextRow |+| centroidRow |+| boxRow

    given csvRowDecoderForMergerAssessedRoi(using 
        CsvRowDecoder[IndexedDetectedSpot, Header], 
        CellDecoder[Set[RoiIndex]], 
    ): CsvRowDecoder[MergerAssessedRoi, Header] = 
        given CsvRowDecoder[Set[RoiIndex], Header] with
            override def apply(row: RowF[Some, Header]): DecoderResult[Set[RoiIndex]] = 
                ColumnNames.MergeRoisColumnName.from(row)
                    .leftMap{ errorMessages => 
                        val context = s"problem(s) decoding merge indices from CSV row ($row)"
                        DecoderError(s"${errorMessages.length} $context: ${errorMessages.mkString_("; ")}")
                    }
                    .toEither
        getCsvRowDecoderForProduct2{ (spot: IndexedDetectedSpot, mergeInputs: Set[RoiIndex]) => 
            MergerAssessedRoi.build(spot, mergeInputs).fold(errMsg => throw new DecoderError(errMsg), identity)
        }

    given csvRowEncoderForMergerAssessedRoi(using 
        encSpot: CsvRowEncoder[IndexedDetectedSpot, Header], 
        encRoiIndices: CellEncoder[Set[RoiIndex]],
    ): CsvRowEncoder[MergerAssessedRoi, Header] = 
        given CsvRowEncoder[Set[RoiIndex], Header] with
            override def apply(elem: Set[RoiIndex]): RowF[Some, Header] = 
                ColumnNames.MergeRoisColumnName.write(elem)
        getCsvRowEncoderForProduct2(_.spot, _.mergeNeighbors)

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

    given csvRowEncoderForMergeContributorRoi(using
        encIndex: CellEncoder[RoiIndex], 
        encContext: CsvRowEncoder[ImagingContext, Header], 
        encCentroid: CsvRowEncoder[Centroid[Double], Header],
        encBox: CsvRowEncoder[BoundingBox, Header],
    ): CsvRowEncoder[MergeContributorRoi, Header] = new:
        override def apply(elem: MergeContributorRoi): RowF[Some, Header] = 
            val originalIndexRow: NamedRow = ColumnNames.RoiIndexColumnName.write(elem.index)
            val contextRow: NamedRow = encContext(elem.context)
            val centroidRow: NamedRow = encCentroid(elem.centroid)
            val boxRow: NamedRow = encBox(elem.box)
            originalIndexRow |+| contextRow |+| centroidRow |+| boxRow

    def parseFromRow[A](messagePrefix: String)(using dec: CsvRowDecoder[A, Header]): RowF[Some, Header] => ValidatedNel[String, A] = 
        row => dec(row)
            .leftMap{ e => s"$messagePrefix: ${e.getMessage}" }
            .toValidatedNel
        
    def parseDetectedSpotRoi(using CsvRowDecoder[DetectedSpotRoi, Header]): RowF[Some, Header] => ValidatedNel[String, DetectedSpotRoi] = 
        parseFromRow("Error decoding ROI")

    given csvRowDecoderForMergedRoiRecord(using 
        decIndex: CellDecoder[RoiIndex], 
        decContext: CsvRowDecoder[ImagingContext, Header], 
        decCentroid: CsvRowDecoder[Centroid[Double], Header], 
        decBox: CsvRowDecoder[BoundingBox, Header], 
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
                val contextNel = decContext(row)
                    .leftMap(e => s"Problem parsing imaging context: ${e.getMessage}")
                    .toValidatedNel
                val centroidNel = decCentroid(row)
                    .leftMap(e => s"Problem parsing centroid: ${e.getMessage}")
                    .toValidatedNel
                val boxNel = decBox(row)
                    .leftMap(e => s"Problem parsing bounding box: ${e.getMessage}")
                    .toValidatedNel
                val contributorsNel = ColumnNames.MergeContributorsColumnName.from(row)
                (indexNel, contextNel, centroidNel, boxNel, contributorsNel)
                    .mapN(MergedRoiRecord.apply)
                    .toEither
                    .leftMap{ messages => DecoderError(s"${messages.length} error(s) decoding merged ROI record: ${messages.mkString_("; ")}") }

    private given cellEncoderForNonemptyRoiBag(
        using encBag: CellEncoder[Set[RoiIndex]]
    ): CellEncoder[NonEmptySet[RoiIndex]] = 
        encBag.contramap((_: NonEmptySet[RoiIndex]).toSortedSet)

    given csvRowEncoderForMergedRoiRecord(using 
        encIndex: CellEncoder[RoiIndex], 
        encContext: CsvRowEncoder[ImagingContext, Header],
        encCentroid: CellEncoder[Double],
        encRoiBag: CellEncoder[Set[RoiIndex]],
    ): CsvRowEncoder[MergedRoiRecord, Header] = 
        val encContribs: CsvRowEncoder[NonEmptySet[RoiIndex], Header] = 
            getCsvRowEncoderForSingleton(ColumnNames.MergeContributorsColumnName)
        new:
            override def apply(elem: MergedRoiRecord): RowF[Some, Header] = 
                val idxRow: NamedRow = ColumnNames.RoiIndexColumnName.write(elem.index)
                val ctxRow: NamedRow = encContext(elem.context)
                val centroidRow: NamedRow = summon[CsvRowEncoder[Centroid[Double], Header]](elem.centroid)
                val contribsRow: NamedRow = encContribs(elem.contributors)
                idxRow |+| ctxRow |+| centroidRow |+| contribsRow

    given csvRowEncoderForUnidentifiableRoi(using
        encIndex: CellEncoder[RoiIndex], 
        encContext: CsvRowEncoder[ImagingContext, Header],
        encCentroid: CellEncoder[Double],
        encBox: CsvRowEncoder[BoundingBox, String],
        encRoiBag: CellEncoder[Set[RoiIndex]],
    ): CsvRowEncoder[UnidentifiableRoi, Header] = 
        val encTooClose: CsvRowEncoder[NonEmptySet[RoiIndex], Header] = 
            getCsvRowEncoderForSingleton(ColumnNames.TooCloseRoisColumnName)
        new:
            override def apply(elem: UnidentifiableRoi): RowF[Some, Header] = 
                val idxRow: NamedRow = ColumnNames.RoiIndexColumnName.write(elem.index)
                val ctxRow: NamedRow = encContext(elem.context)
                val centroidRow: NamedRow = summon[CsvRowEncoder[Centroid[Double], Header]](elem.centroid)
                val boxRow: NamedRow = encBox(elem.box)
                val tooCloseRow: NamedRow = encTooClose(elem.tooClose)
                idxRow |+| ctxRow |+| centroidRow |+| boxRow |+| tooCloseRow

    
    given decoderForPostMergeRoi(using 
        decIndex: CellDecoder[RoiIndex],
        decSpot: CsvRowDecoder[IndexedDetectedSpot, String], 
    ): CsvRowDecoder[PostMergeRoi, String] = 
        given Ordering[RoiIndex] = summon[Order[RoiIndex]].toOrdering
        new:
            override def apply(row: RowF[Some, String]): DecoderResult[PostMergeRoi] = 
                val spotNel = parseFromRow[IndexedDetectedSpot]("Error(s) decoding ROI")(row)
                val contributorsNel = ColumnNames.MergeRoisColumnName.from(row)
                (spotNel, contributorsNel)
                    .tupled
                    .toEither
                    .map{ (spot, contributors) => 
                        contributors.toNonEmptySet.fold(spot){ 
                            contribs => MergedRoiRecord(spot, contribs) 
                        }
                    }
                    .leftMap{ messages => 
                        DecoderError(s"${messages.length} error(s) decoding ROI: ${messages.mkString_("; ")}") 
                    }

    given encoderForPostMergeRoi(using 
        encIdx: CellEncoder[RoiIndex],
        encContext: CsvRowEncoder[ImagingContext, String], 
        encCentroid: CsvRowEncoder[Centroid[Double], String], 
        encBox: CsvRowEncoder[BoundingBox, String],
    ): CsvRowEncoder[PostMergeRoi, String] = 
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
                    case merged: MergedRoiRecord => merged.contributors.toSortedSet
                }
                val mergeRoisRow: NamedRow = ColumnNames.MergeRoisColumnName.write(contribs)
                idxRow |+| ctxRow |+| encCentroid(centroid) |+| encBox(box) |+| mergeRoisRow
end RoiCsvInstances
