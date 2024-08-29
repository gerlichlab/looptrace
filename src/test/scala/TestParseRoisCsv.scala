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
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{
    getCsvRowDecoderForProduct2,
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

import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.space.{BoundingBox as BB, *}

/**
  * Tests for parsing different types of ROIs.
  */
class TestParseRoisCsv extends AnyFunSuite, LooptraceSuite, should.Matchers, ScalaCheckPropertyChecks:

    type HeaderField = String
    type RawCoordinate = Double

    test("Small detected spot ROI example parses correctly.") {
        val linesToWrite = 
            """
            ,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,nucleusNumber
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,0
            1,P0001.zarr,80,2,17.994259347453493,24.042015416774795,1360.0069098862991,213.58943029032,117.1394688491732,11.994259347453491,23.994259347453493,12.042015416774795,36.0420154167748,1348.0069098862991,1372.0069098862991,0
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
                        if (unequal.nonEmpty) {
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
            ",fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax",
            "fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax",
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

    test("Small ProximityAssessedRoi example parses correctly.") {
        val linesToWrite = """
            index,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,tooCloseRois,mergeRois
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,4;3,10
            1,P0001.zarr,80,2,17.994259347453493,24.042015416774795,1360.0069098862991,213.58943029032,117.13946884917321,11.994259347453493,23.994259347453493,12.042015416774795,36.042015416774795,1348.0069098862991,1372.0069098862991,5,7;8
            """.toLines

        val expectedRecords: List[ProximityAssessedRoi] = {
            val pos = PositionName("P0001.zarr")

            val rec1: ProximityAssessedRoi = {
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
                ProximityAssessedRoi.build(
                    RoiIndex(0), 
                    DetectedSpotRoi(spot, box), 
                    tooClose = Set(3, 4).map(RoiIndex.unsafe),
                    merge = Set(10).map(RoiIndex.unsafe),
                )
                .fold(errors => throw new Exception(s"${errors.length} error(s) building ROI example."), identity)
            }

            val rec2: ProximityAssessedRoi = {
                val xBounds = XCoordinate(1348.0069098862991) -> XCoordinate(1372.0069098862991)
                val yBounds = YCoordinate(12.042015416774795) -> YCoordinate(36.042015416774795)
                val zBounds = ZCoordinate(11.994259347453493) -> ZCoordinate(23.994259347453493)
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
                    MeanIntensity(117.13946884917321),
                )
                val box = BoundingBox(
                    BoundingBox.Interval.unsafeFromTuple(xBounds), 
                    BoundingBox.Interval.unsafeFromTuple(yBounds), 
                    BoundingBox.Interval.unsafeFromTuple(zBounds),
                )
                ProximityAssessedRoi.build(
                    RoiIndex(1), 
                    DetectedSpotRoi(spot, box), 
                    tooClose = Set(5).map(RoiIndex.unsafe),
                    merge = Set(7, 8).map(RoiIndex.unsafe),
                ).fold(errors => throw new Exception(s"${errors.length} error(s) building ROI example."), identity)
            }

            List(rec1, rec2)
        }
        
        withTempFile(linesToWrite, Delimiter.CommaSeparator){ roisFile =>
            val observedRecords: List[ProximityAssessedRoi] = unsafeRead(roisFile)
            observedRecords.length shouldEqual expectedRecords.length // quick, simplifying check
            observedRecords.map(_.roi.spot) shouldEqual expectedRecords.map(_.roi.spot)
            observedRecords shouldEqual expectedRecords // full check
        }
    }

    test("ProximityAssessedRoi records roundtrip through CSV.") {
        // Generate legal combination of main ROI index, too-close ROIs, and ROIs to merge.
        def genRoiIndexAndRoiBags(using Arbitrary[RoiIndex]): Gen[(RoiIndex, (Set[RoiIndex], Set[RoiIndex]))] = 
            for {
                idx <- Arbitrary.arbitrary[RoiIndex]
                raw1 <- Gen.listOf(Arbitrary.arbitrary[RoiIndex])
                bag1 = raw1.toSet - idx // Prevent overlap with the main index
                raw2 <- Gen.listOf(Arbitrary.arbitrary[RoiIndex])
                bag2 = (raw2.toSet -- bag1) - idx // Prevent overlap with other bag and with main index.
            } yield (idx, (bag1, bag2))

        given arbRoi(using Arbitrary[RoiIndex], Arbitrary[DetectedSpotRoi]): Arbitrary[ProximityAssessedRoi] = 
            Arbitrary{
                for {
                    roi <- Arbitrary.arbitrary[DetectedSpotRoi]
                    (index, (tooClose, forMerge)) <- genRoiIndexAndRoiBags
                } yield ProximityAssessedRoi
                    .build(index, roi, tooClose, forMerge)
                    .fold(errors => throw new Exception(s"ROI build error(s): $errors"), identity)
            }

        forAll { (inputRecords: NonEmptyList[ProximityAssessedRoi]) => 
            withTempFile(Delimiter.CommaSeparator){ roisFile =>
                /* First, write the records to CSV */
                val sink: Pipe[IO, ProximityAssessedRoi, Nothing] = 
                    writeCaseClassesToCsv[ProximityAssessedRoi](roisFile)
                Stream.emits(inputRecords.toList).through(sink).compile.drain.unsafeRunSync()

                /* Then, do the parse-and-check. */
                Try{ unsafeRead[ProximityAssessedRoi](roisFile) } match {
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
                        if (unequal.nonEmpty) {
                            fail(s"Unequal records pairs (below):\n${unequal.map{ (in, out) => in.toString ++ "\n" ++ out.toString }.mkString("\n\n")}")
                        } else {
                            succeed
                        }
                }
            }
        }
    }

    test("Header-only file gives empty list of results for ProximityAssessedRoi.") {
        val headers = List(
            "index,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,tooCloseRois,mergeRois",
        )
        val newlines = List(false, true)
        val grid = headers.flatMap(h => newlines.map(p => h -> p))
        
        forAll (Table(("header", "newline"), grid*)) { (header, newline) => 
            val fileData = header ++ (if newline then "\n" else "")
            withTempFile(fileData, Delimiter.CommaSeparator) { roisFile => 
                val expected = List.empty[ProximityAssessedRoi]
                val observed = unsafeRead[ProximityAssessedRoi](roisFile)
                observed shouldEqual expected
            }
        }
    }

    test("ProximityAssessedRoi cannot be parsed from pandas-style no-name index column.") {
        val initData = """
            ,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,tooCloseRois,mergeRois
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,,
            """.toLines
        
        withTempFile(initData, Delimiter.CommaSeparator) { roisFile => 
            assertThrows[DecoderError]{ unsafeRead[ProximityAssessedRoi](roisFile) }
        }
    }

    test("ProximityAssessedRoi cannot be parsed when either proximal ROIs column is missing.") {
        val datas = List(
            """
            index,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,tooCloseRois
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,
            """.toLines,
            """
            index,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,mergeRois
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,
            """.toLines,
            """
            index,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726
            """.toLines,
        )

        forAll (Table(("initData"), datas*)) { initData => 
            withTempFile(initData, Delimiter.CommaSeparator) { roisFile => 
                assertThrows[DecoderError]{ unsafeRead[ProximityAssessedRoi](roisFile) }
            }
        }
    }

    private def unsafeRead[A](roisFile: os.Path)(using CsvRowDecoder[A, String], CharLikeChunks[IO, Byte]): List[A] = 
        readCsvToCaseClasses[A](roisFile).unsafeRunSync()

    extension (example: String)
        // Utility function to trim line endings and whitespace, accounting for formatting of raw example data.
        private def toLines: List[String] = example.split("\n").map(_.trim).filterNot(_.isEmpty).toList.map(_ ++ "\n")
end TestParseRoisCsv
