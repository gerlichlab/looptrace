package at.ac.oeaw.imba.gerlich.looptrace
package csv

import scala.util.{ Failure, Success, Try }

import cats.data.NonEmptyList
import cats.effect.IO
import cats.effect.unsafe.implicits.global // needed to call unsafeRunSync()
import cats.syntax.all.*
import fs2.{ Pipe, Stream }
import fs2.data.csv.*
import fs2.data.text.CharLikeChunks
import fs2.data.text.utf8.byteStreamCharLike // to prove CharLikeChunks[cats.effect.IO, Byte]

import org.scalacheck.*
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import io.github.iltotore.iron.{ autoCastIron, autoRefine }
import io.github.iltotore.iron.scalacheck.all.given

import at.ac.oeaw.imba.gerlich.gerlib.geometry.BoundingBox
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid

import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.SpotChannelColumnName
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{
    getCsvRowEncoderForProduct2,
    readCsvToCaseClasses, 
    writeCaseClassesToCsv, 
}
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.{
    getCsvRowDecoderForNuclearDesignation,
    getCsvRowEncoderForNuclearDesignation,
    given,
}
import at.ac.oeaw.imba.gerlich.gerlib.roi.DetectedSpot
import at.ac.oeaw.imba.gerlich.gerlib.roi.measurement.{ Area, MeanIntensity }
import at.ac.oeaw.imba.gerlich.gerlib.testing.instances.all.given

import at.ac.oeaw.imba.gerlich.looptrace.csv.getCsvRowDecoderForImagingChannel
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.roi.{
    DetectedSpotRoi,
    MergerAssessedRoi,
}
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.IndexedDetectedSpot
import at.ac.oeaw.imba.gerlich.looptrace.space.{BoundingBox as BB, *}

/**
  * Tests for parsing different types of ROIs.
  */
class TestParseRoisCsv extends AnyFunSuite, LooptraceSuite, should.Matchers, ScalaCheckPropertyChecks:

    type HeaderField = String

    test("Small detected spot ROI example parses correctly.") {
        val linesToWrite = 
            """
            index,fieldOfView,timepoint,spotChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726
            1,P0001.zarr,80,2,17.994259347453493,24.042015416774795,1360.0069098862991,213.58943029032,117.1394688491732,11.994259347453491,23.994259347453493,12.042015416774795,36.0420154167748,1348.0069098862991,1372.0069098862991
            """.toLines
    
        val expectedRecords: List[DetectedSpotRoi] = {
            val pos = PositionName("P0001.zarr")

            val rec1: DetectedSpotRoi = {
                val xBounds = XCoordinate(859.9833511648726) -> XCoordinate(883.9833511648726)
                val yBounds = YCoordinate(219.9874778925304) -> YCoordinate(243.9874778925304)
                val zBounds = ZCoordinate(-2.092371012467521) -> ZCoordinate(9.90762898753248)
                val spot = DetectedSpot(
                    ImagingContext(
                        pos,
                        ImagingTimepoint(79),
                        ImagingChannel(0),
                    ),
                    Centroid.fromPoint(
                        Point3D(
                            XCoordinate(871.9833511648726),
                            YCoordinate(231.9874778925304),
                            ZCoordinate(3.907628987532479),
                        ), 
                    ),
                    Area(240.00390423),
                    MeanIntensity(118.26726920593931),
                )
                val box = BoundingBox(
                    BoundingBox.Interval.unsafeFromTuple(xBounds), 
                    BoundingBox.Interval.unsafeFromTuple(yBounds), 
                    BoundingBox.Interval.unsafeFromTuple(zBounds),
                )
                DetectedSpotRoi(spot, box)
            }

            val rec2: DetectedSpotRoi = {
                val xBounds = XCoordinate(1348.0069098862991) -> XCoordinate(1372.0069098862991)
                val yBounds = YCoordinate(12.042015416774795) -> YCoordinate(36.0420154167748)
                val zBounds = ZCoordinate(11.994259347453491) -> ZCoordinate(23.994259347453493)
                val spot = DetectedSpot(
                    ImagingContext(
                        pos,
                        ImagingTimepoint(80),
                        ImagingChannel(2),
                    ),
                    Centroid.fromPoint(
                        Point3D(
                            XCoordinate(1360.0069098862991),
                            YCoordinate(24.042015416774795),
                            ZCoordinate(17.994259347453493),
                        ), 
                    ),
                    Area(213.58943029032),
                    MeanIntensity(117.1394688491732),
                )
                val box = BoundingBox(
                    BoundingBox.Interval.unsafeFromTuple(xBounds), 
                    BoundingBox.Interval.unsafeFromTuple(yBounds), 
                    BoundingBox.Interval.unsafeFromTuple(zBounds),
                )
                DetectedSpotRoi(spot, box)
            }

            List(rec1, rec2)
        }

        withTempFile(linesToWrite, Delimiter.CommaSeparator){ roisFile =>
            val observedRecords: List[DetectedSpotRoi] = unsafeRead(roisFile)
            observedRecords.length shouldEqual expectedRecords.length // quick, simplifying check
            observedRecords shouldEqual expectedRecords // full check
        }
    }
    
    test("DetectedSpotRoi records roundtrip through CSV.") {
        forAll { (inputRecords: NonEmptyList[DetectedSpotRoi]) => 
            withTempFile(Delimiter.CommaSeparator){ roisFile =>
                /* First, write the records to CSV */
                val sink: Pipe[IO, DetectedSpotRoi, Nothing] = writeCaseClassesToCsv[DetectedSpotRoi](roisFile)
                Stream.emits(inputRecords.toList).through(sink).compile.drain.unsafeRunSync()

                /* Then, do the parse-and-check. */
                Try{ unsafeRead[DetectedSpotRoi](roisFile) } match {
                    case Failure(e) => 
                        println("LINES (below):")
                        os.read.lines(roisFile).foreach(println)
                        fail(s"Expected good parse but got error: $e")
                    case Success(outputRecords) => 
                        /* quick checks for proper record count */
                        outputRecords.nonEmpty shouldBe true
                        outputRecords.length shouldEqual inputRecords.length
                        // Check the actual equality, element-by-element.-
                        val unequal = inputRecords.toList.zip(outputRecords).filter{ (in, out) => in != out }
                        if unequal.nonEmpty then {
                            fail(s"Unequal records pairs (below):\n${unequal.map{ (in, out) => in.toString ++ "\n" ++ out.toString }.mkString("\n\n")}")
                        } else {
                            succeed
                        }
                }
            }
        }
    }

    test("Header-only file gives empty list of results for DetectedSpotRoi.") {
        val headers = List(
            "index,fieldOfView,timepoint,spotChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax",
            "index,fieldOfView,timepoint,spotChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax",
        )
        val newlines = List(false, true)
        val grid = headers.flatMap(h => newlines.map(p => h -> p))
        forAll (Table(("header", "newline"), grid*)) { (header, newline) => 
            val fileData = header ++ (if newline then "\n" else "")
            withTempFile(fileData, Delimiter.CommaSeparator) { roisFile => 
                val expected = List.empty[DetectedSpotRoi]
                val observed = unsafeRead[DetectedSpotRoi](roisFile)
                observed shouldEqual expected
            }
        }
    }

    test("Small MergerAssessedRoi example parses correctly.") {
        val linesToWrite = """
            index,fieldOfView,timepoint,spotChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,mergePartners
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,10
            1,P0001.zarr,80,2,17.994259347453493,24.042015416774795,1360.0069098862991,213.58943029032,117.13946884917321,11.994259347453493,23.994259347453493,12.042015416774795,36.042015416774795,1348.0069098862991,1372.0069098862991,7 8
            """.toLines

        val expectedRecords: List[MergerAssessedRoi] = {
            val pos = PositionName("P0001.zarr")

            val rec1: MergerAssessedRoi = {
                val xBounds = XCoordinate(859.9833511648726) -> XCoordinate(883.9833511648726)
                val yBounds = YCoordinate(219.9874778925304) -> YCoordinate(243.9874778925304)
                val zBounds = ZCoordinate(-2.092371012467521) -> ZCoordinate(9.90762898753248)
                val box = BoundingBox(
                    BoundingBox.Interval.unsafeFromTuple(xBounds), 
                    BoundingBox.Interval.unsafeFromTuple(yBounds), 
                    BoundingBox.Interval.unsafeFromTuple(zBounds),
                )
                
                val spot = IndexedDetectedSpot(
                    RoiIndex(0), 
                    ImagingContext(
                        pos,
                        ImagingTimepoint(79),
                        ImagingChannel(0),
                    ),
                    Centroid.fromPoint(
                        Point3D(
                            XCoordinate(871.9833511648726),
                            YCoordinate(231.9874778925304),
                            ZCoordinate(3.907628987532479),
                        ), 
                    ),
                    box,
                    // Area(240.00390423),
                    // MeanIntensity(118.26726920593931),
                )
                MergerAssessedRoi.build(
                    spot,
                    Set(10).map(RoiIndex.unsafe),
                )
                .fold(errors => throw new Exception(s"${errors.length} error(s) building ROI example."), identity)
            }

            val rec2: MergerAssessedRoi = {
                val xBounds = XCoordinate(1348.0069098862991) -> XCoordinate(1372.0069098862991)
                val yBounds = YCoordinate(12.042015416774795) -> YCoordinate(36.042015416774795)
                val zBounds = ZCoordinate(11.994259347453493) -> ZCoordinate(23.994259347453493)
                val box = BoundingBox(
                    BoundingBox.Interval.unsafeFromTuple(xBounds), 
                    BoundingBox.Interval.unsafeFromTuple(yBounds), 
                    BoundingBox.Interval.unsafeFromTuple(zBounds),
                )
                val spot = IndexedDetectedSpot(
                    RoiIndex(1), 
                    ImagingContext(
                        pos,
                        ImagingTimepoint(80),
                        ImagingChannel(2),
                    ),
                    Centroid.fromPoint(
                        Point3D(
                            XCoordinate(1360.0069098862991),
                            YCoordinate(24.042015416774795),
                            ZCoordinate(17.994259347453493),
                        ), 
                    ),
                    box, 
                    // Area(213.58943029032),
                    // MeanIntensity(117.13946884917321),
                )
                MergerAssessedRoi.build(
                    spot,
                    Set(7, 8).map(RoiIndex.unsafe),
                ).fold(errors => throw new Exception(s"${errors.length} error(s) building ROI example."), identity)
            }

            List(rec1, rec2)
        }
        
        withTempFile(linesToWrite, Delimiter.CommaSeparator){ roisFile =>
            val observedRecords: List[MergerAssessedRoi] = 
                given CsvRowDecoder[ImagingChannel, String] = getCsvRowDecoderForImagingChannel(SpotChannelColumnName)
                unsafeRead(roisFile)
            observedRecords.length shouldEqual expectedRecords.length // quick, simplifying check
            observedRecords.map(_.spot) shouldEqual expectedRecords.map(_.spot)
            observedRecords shouldEqual expectedRecords // full check
        }
    }

    test("MergerAssessedRoi records roundtrip through CSV.") {
        // Generate legal combination of main ROI index, too-close ROIs, and ROIs to merge.
        def genRoiIndexAndMergeIndices(using Arbitrary[RoiIndex]): Gen[(RoiIndex, Set[RoiIndex])] = 
            for
                idx <- Arbitrary.arbitrary[RoiIndex]
                forMergeRaw <- Gen.listOf(Arbitrary.arbitrary[RoiIndex])
            yield (idx, forMergeRaw.toSet - idx)

        given (Arbitrary[RoiIndex], Arbitrary[IndexedDetectedSpot]) => Arbitrary[MergerAssessedRoi] = 
            Arbitrary:
                for
                    roi <- Arbitrary.arbitrary[IndexedDetectedSpot]
                    (index, forMerge) <- genRoiIndexAndMergeIndices
                yield MergerAssessedRoi
                    .build(roi.copy(index = index), forMerge)
                    .fold(errors => throw new Exception(s"ROI build error(s): $errors"), identity)

        forAll { (inputRecords: NonEmptyList[MergerAssessedRoi]) => 
            given CsvRowDecoder[ImagingChannel, String] = getCsvRowDecoderForImagingChannel(SpotChannelColumnName)
            given CsvRowEncoder[ImagingChannel, String] = SpotChannelColumnName.toNamedEncoder
            withTempFile(Delimiter.CommaSeparator){ roisFile =>
                /* First, write the records to CSV */
                val sink: Pipe[IO, MergerAssessedRoi, Nothing] = 
                    writeCaseClassesToCsv[MergerAssessedRoi](roisFile)
                Stream.emits(inputRecords.toList).through(sink).compile.drain.unsafeRunSync()

                /* Then, do the parse-and-check. */
                Try{ unsafeRead[MergerAssessedRoi](roisFile) } match {
                    case Failure(e) => 
                        println("LINES (below):")
                        os.read.lines(roisFile).foreach(println)
                        fail(s"Expected good parse but got error: $e")
                    case Success(outputRecords) => 
                        /* quick checks for proper record count */
                        outputRecords.nonEmpty shouldBe true
                        outputRecords.length shouldEqual inputRecords.length
                        // Check the actual equality, element-by-element.-
                        val unequal = inputRecords.toList.zip(outputRecords).filter{ (in, out) => in != out }
                        if unequal.nonEmpty then {
                            fail(s"Unequal records pairs (below):\n${unequal.map{ (in, out) => in.toString ++ "\n" ++ out.toString }.mkString("\n\n")}")
                        } else {
                            succeed
                        }
                }
            }
        }
    }

    test("Header-only file gives empty list of results for MergerAssessedRoi.") {
        val headers = List(
            "index,fieldOfView,timepoint,spotChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,mergePartners",
        )
        val newlines = List(false, true)
        val grid = headers.flatMap(h => newlines.map(p => h -> p))
        
        forAll (Table(("header", "newline"), grid*)) { (header, newline) => 
            val fileData = header ++ (if newline then "\n" else "")
            withTempFile(fileData, Delimiter.CommaSeparator) { roisFile => 
                val expected = List.empty[MergerAssessedRoi]
                val observed = 
                    given CsvRowDecoder[ImagingChannel, String] = 
                        getCsvRowDecoderForImagingChannel(SpotChannelColumnName)
                    unsafeRead[MergerAssessedRoi](roisFile)
                observed shouldEqual expected
            }
        }
    }

    test("MergerAssessedRoi cannot be parsed from pandas-style no-name index column.") {
        val initData = """
            ,fieldOfView,timepoint,spotChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,mergePartners
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,
            """.toLines
        
        given CsvRowDecoder[ImagingChannel, String] = 
            getCsvRowDecoderForImagingChannel(SpotChannelColumnName)

        withTempFile(initData, Delimiter.CommaSeparator) { roisFile => 
            assertThrows[DecoderError]{ unsafeRead[MergerAssessedRoi](roisFile) }
        }
    }

    test("MergerAssessedRoi cannot be parsed when either merger ROIs column is missing.") {
        val datas = List(
            """
            index,fieldOfView,timepoint,spotChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726
            """.toLines,
        )

        given CsvRowDecoder[ImagingChannel, String] = 
            getCsvRowDecoderForImagingChannel(SpotChannelColumnName)

        forAll (Table(("initData"), datas*)) { initData => 
            withTempFile(initData, Delimiter.CommaSeparator) { roisFile => 
                assertThrows[DecoderError]{ unsafeRead[MergerAssessedRoi](roisFile) }
            }
        }
    }

    private def unsafeRead[A](roisFile: os.Path)(using CsvRowDecoder[A, String], CharLikeChunks[IO, Byte]): List[A] = 
        readCsvToCaseClasses[A](roisFile).unsafeRunSync()

    extension (example: String)
        // Utility function to trim line endings and whitespace, accounting for formatting of raw example data.
        private def toLines: List[String] = example.split("\n").map(_.trim).filterNot(_.isEmpty).toList.map(_ ++ "\n")
end TestParseRoisCsv
