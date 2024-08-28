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

import at.ac.oeaw.imba.gerlich.gerlib.cell.{ NuclearDesignation, NucleusNumber, OutsideNucleus }
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

    test("Small NucleusLabelAttemptedRoi example parses correctly.") {
        val linesToWrite = 
            """
            ,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,nucleusNumber
            0,P0001.zarr,9,1,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,2
            1,P0001.zarr,10,0,17.994259347453493,24.042015416774795,1360.0069098862991,213.58943029032,117.1394688491732,11.994259347453491,23.994259347453493,12.042015416774795,36.0420154167748,1348.0069098862991,1372.0069098862991,0
            2,P0001.zarr,10,2,23.00910242218976,231.98008711401275,871.9596645390719,226.90422978,116.14075915047448,17.009102422189756,29.00910242218976,219.98008711401275,243.98008711401275,859.9596645390719,883.9596645390719,5
            """.toLines

        val pos = PositionName("P0001.zarr")

        val expectedRecords: List[NucleusLabelAttemptedRoi] = {
            val rec1: NucleusLabelAttemptedRoi = {
                val xBounds = XCoordinate(859.9833511648726) -> XCoordinate(883.9833511648726)
                val yBounds = YCoordinate(219.9874778925304) -> YCoordinate(243.9874778925304)
                val zBounds = ZCoordinate(-2.092371012467521) -> ZCoordinate(9.90762898753248)
                val spot = DetectedSpot(
                    ImagingContext(
                        pos,
                        ImagingTimepoint(9),
                        ImagingChannel(1),
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
                NucleusLabelAttemptedRoi(spot, box, NucleusNumber(2))
            }

            val rec2: NucleusLabelAttemptedRoi = {
                val xBounds = XCoordinate(1348.0069098862991) -> XCoordinate(1372.0069098862991)
                val yBounds = YCoordinate(12.042015416774795) -> YCoordinate(36.0420154167748)
                val zBounds = ZCoordinate(11.994259347453491) -> ZCoordinate(23.994259347453493)
                val spot = DetectedSpot(
                    ImagingContext(
                        pos,
                        ImagingTimepoint(10),
                        ImagingChannel(0),
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
                NucleusLabelAttemptedRoi(spot, box, OutsideNucleus)
            }

            val rec3: NucleusLabelAttemptedRoi = {
                val xBounds = XCoordinate(859.9596645390719) -> XCoordinate(883.9596645390719)
                val yBounds = YCoordinate(219.98008711401275) -> YCoordinate(243.98008711401275)
                val zBounds = ZCoordinate(17.009102422189756) -> ZCoordinate(29.00910242218976)
                val spot = DetectedSpot(
                    ImagingContext(
                        pos,
                        ImagingTimepoint(10),
                        ImagingChannel(2),
                    ),
                    Centroid.fromPoint(
                        Point3D(
                            XCoordinate(871.9596645390719),
                            YCoordinate(231.98008711401275),
                            ZCoordinate(23.00910242218976),
                        ), 
                    ),
                    Area(226.90422978),
                    MeanIntensity(116.14075915047448),
                )
                val box = BoundingBox(
                    BoundingBox.Interval.unsafeFromTuple(xBounds), 
                    BoundingBox.Interval.unsafeFromTuple(yBounds), 
                    BoundingBox.Interval.unsafeFromTuple(zBounds),
                )
                NucleusLabelAttemptedRoi(spot, box, NucleusNumber(5))
            }

            List(rec1, rec2, rec3)
        }

        // Additional component to CsvRowDecoder[DetectedSpotRoi, HeaderField] to derive 
        // CsvRowDecoder[NucleusLabelAttemptedRoi, HeaderField]
        given CsvRowDecoder[NuclearDesignation, HeaderField] = getCsvRowDecoderForNuclearDesignation()
        
        withTempFile(linesToWrite, Delimiter.CommaSeparator){ roisFile =>
            val observedRecords: List[NucleusLabelAttemptedRoi] = unsafeRead(roisFile)
            observedRecords.length shouldEqual expectedRecords.length // quick, simplifying check
            observedRecords shouldEqual expectedRecords // full check
        }
    }

    test("NucleusLabelAttemptedRoi records roundtrip through CSV.") {
        // Additional component to CsvRowDecoder[DetectedSpotRoi, HeaderField] to derive 
        // CsvRowDecoder[NucleusLabelAttemptedRoi, HeaderField]
        given CsvRowDecoder[NuclearDesignation, HeaderField] = getCsvRowDecoderForNuclearDesignation()
        // Additional component to CsvRowEncoder[DetectedSpotRoi, HeaderField] to derive 
        // CsvRowEncoder[NucleusLabelAttemptedRoi, HeaderField]
        given CsvRowEncoder[NuclearDesignation, HeaderField] = getCsvRowEncoderForNuclearDesignation()

        forAll { (inputRecords: NonEmptyList[NucleusLabelAttemptedRoi]) => 
            withTempFile(Delimiter.CommaSeparator){ roisFile =>
                /* First, write the records to CSV */
                val sink: Pipe[IO, NucleusLabelAttemptedRoi, Nothing] = writeCaseClassesToCsv[NucleusLabelAttemptedRoi](roisFile)
                Stream.emits(inputRecords.toList).through(sink).compile.drain.unsafeRunSync()

                /* Then, do the parse-and-check. */
                Try{ unsafeRead[NucleusLabelAttemptedRoi](roisFile) } match {
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

    test("Header-only file gives empty list of results for NucleusLabelAttemptedRoi.") {
        val headers = List(
            ",fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,nucleusNumber",
            "fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,nucleusNumber",
        )
        val newlines = List(false, true)
        val grid = headers.flatMap(h => newlines.map(p => h -> p))
        
        // Additional component to CsvRowDecoder[DetectedSpotRoi, HeaderField] to derive 
        // CsvRowDecoder[NucleusLabelAttemptedRoi, HeaderField]
        given CsvRowDecoder[NuclearDesignation, HeaderField] = getCsvRowDecoderForNuclearDesignation()
        
        forAll (Table(("header", "newline"), grid*)) { (header, newline) => 
            val fileData = header ++ (if newline then "\n" else "")
            withTempFile(fileData, Delimiter.CommaSeparator) { roisFile => 
                val expected = List.empty[NucleusLabelAttemptedRoi]
                val observed = unsafeRead[NucleusLabelAttemptedRoi](roisFile)
                observed shouldEqual expected
            }
        }
    }

    test("Small NucleusLabeledProximityAssessedRoi example parses correctly.") {
        val linesToWrite = """
            index,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,nucleusNumber,tooCloseRois,mergeRois,groupingRois
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,2,4;3,10,9;8;7
            1,P0001.zarr,80,2,17.994259347453493,24.042015416774795,1360.0069098862991,213.58943029032,117.13946884917321,11.994259347453493,23.994259347453493,12.042015416774795,36.042015416774795,1348.0069098862991,1372.0069098862991,0,5,7;8,4;6
            """.toLines

        val expectedRecords: List[NucleusLabeledProximityAssessedRoi] = {
            val pos = PositionName("P0001.zarr")

            val rec1: NucleusLabeledProximityAssessedRoi = {
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
                NucleusLabeledProximityAssessedRoi.build(
                    RoiIndex(0), 
                    DetectedSpotRoi(spot, box), 
                    NucleusNumber(2),
                    tooClose = Set(3, 4).map(RoiIndex.unsafe),
                    merge = Set(10).map(RoiIndex.unsafe),
                    groupForAnalysis = Set(7, 8, 9).map(RoiIndex.unsafe)
                )
                .fold(errors => throw new Exception(s"${errors.length} error(s) building ROI example."), identity)
            }

            val rec2: NucleusLabeledProximityAssessedRoi = {
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
                NucleusLabeledProximityAssessedRoi.build(
                    RoiIndex(1), 
                    DetectedSpotRoi(spot, box), 
                    OutsideNucleus,
                    tooClose = Set(5).map(RoiIndex.unsafe),
                    merge = Set(7, 8).map(RoiIndex.unsafe),
                    groupForAnalysis = Set(4, 6).map(RoiIndex.unsafe)
                ).fold(errors => throw new Exception(s"${errors.length} error(s) building ROI example."), identity)
            }

            List(rec1, rec2)
        }
        
        // Additional component to CsvRowDecoder[DetectedSpotRoi, HeaderField] to derive 
        // CsvRowDecoder[NucleusLabeledProximityAssessedRoi, HeaderField]
        given CsvRowDecoder[NuclearDesignation, HeaderField] = getCsvRowDecoderForNuclearDesignation()

        withTempFile(linesToWrite, Delimiter.CommaSeparator){ roisFile =>
            val observedRecords: List[NucleusLabeledProximityAssessedRoi] = unsafeRead(roisFile)
            observedRecords.length shouldEqual expectedRecords.length // quick, simplifying check
            observedRecords.map(_.roi.spot) shouldEqual expectedRecords.map(_.roi.spot)
            observedRecords shouldEqual expectedRecords // full check
        }
    }

    test("NucleusLabeledProximityAssessedRoi records roundtrip through CSV.") {
        // Generate legal combination of main ROI index, too-close ROIs, and ROIs to merge.
        def genRoiIndexAndRoiBags(using Arbitrary[RoiIndex]): Gen[(RoiIndex, (Set[RoiIndex], Set[RoiIndex], Set[RoiIndex]))] = 
            for {
                idx <- Arbitrary.arbitrary[RoiIndex]
                raw1 <- Gen.listOf(Arbitrary.arbitrary[RoiIndex])
                bag1 = raw1.toSet - idx // Prevent overlap with the main index
                raw2 <- Gen.listOf(Arbitrary.arbitrary[RoiIndex])
                bag2 = (raw2.toSet -- bag1) - idx // Prevent overlap with other bag and with main index.
                raw3 <- Gen.listOf(Arbitrary.arbitrary[RoiIndex])
                bag3 = ((raw3.toSet -- bag2) -- bag1) - idx
            } yield (idx, (bag1, bag2, bag3))

        given arbRoi(using Arbitrary[RoiIndex], Arbitrary[NucleusLabelAttemptedRoi]): Arbitrary[NucleusLabeledProximityAssessedRoi] = 
            Arbitrary{
                for {
                    roi <- Arbitrary.arbitrary[NucleusLabelAttemptedRoi]
                    (index, (tooClose, forMerge, forGroup)) <- genRoiIndexAndRoiBags
                } yield NucleusLabeledProximityAssessedRoi
                    .build(index, roi.toDetectedSpotRoi, roi.nucleus, tooClose, forMerge, forGroup)
                    .fold(errors => throw new Exception(s"ROI build error(s): $errors"), identity)
            }

        // Additional component to CsvRowDecoder[DetectedSpotRoi, HeaderField] to derive 
        // CsvRowDecoder[NucleusLabeledProximityAssessedRoi, HeaderField]
        given CsvRowDecoder[NuclearDesignation, HeaderField] = getCsvRowDecoderForNuclearDesignation()
        // Additional component to CsvRowEncoder[DetectedSpotRoi, HeaderField] to derive 
        // CsvRowEncoder[NucleusLabeledProximityAssessedRoi, HeaderField]
        given CsvRowEncoder[NuclearDesignation, HeaderField] = getCsvRowEncoderForNuclearDesignation()

        forAll { (inputRecords: NonEmptyList[NucleusLabeledProximityAssessedRoi]) => 
            withTempFile(Delimiter.CommaSeparator){ roisFile =>
                /* First, write the records to CSV */
                val sink: Pipe[IO, NucleusLabeledProximityAssessedRoi, Nothing] = 
                    writeCaseClassesToCsv[NucleusLabeledProximityAssessedRoi](roisFile)
                Stream.emits(inputRecords.toList).through(sink).compile.drain.unsafeRunSync()

                /* Then, do the parse-and-check. */
                Try{ unsafeRead[NucleusLabeledProximityAssessedRoi](roisFile) } match {
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

    test("Header-only file gives empty list of results for NucleusLabeledProximityAssessedRoi.") {
        val headers = List(
            "index,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,nucleusNumber,tooCloseRois,mergeRois",
        )
        val newlines = List(false, true)
        val grid = headers.flatMap(h => newlines.map(p => h -> p))
        
        // Additional component to CsvRowDecoder[DetectedSpotRoi, HeaderField] to derive 
        // CsvRowDecoder[NucleusLabeledProximityAssessedRoi, HeaderField]
        given CsvRowDecoder[NuclearDesignation, HeaderField] = getCsvRowDecoderForNuclearDesignation()
        
        forAll (Table(("header", "newline"), grid*)) { (header, newline) => 
            val fileData = header ++ (if newline then "\n" else "")
            withTempFile(fileData, Delimiter.CommaSeparator) { roisFile => 
                val expected = List.empty[NucleusLabelAttemptedRoi]
                val observed = unsafeRead[NucleusLabelAttemptedRoi](roisFile)
                observed shouldEqual expected
            }
        }
    }

    test("NucleusLabeledProximityAssessedRoi cannot be parsed from pandas-style no-name index column.") {
        val initData = """
            ,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,nucleusNumber,tooCloseRois,mergeRois
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,2,,
            """.toLines
        
        // Additional component to CsvRowDecoder[DetectedSpotRoi, HeaderField] to derive 
        // CsvRowDecoder[NucleusLabeledProximityAssessedRoi, HeaderField]
        given CsvRowDecoder[NuclearDesignation, HeaderField] = getCsvRowDecoderForNuclearDesignation()
        
        withTempFile(initData, Delimiter.CommaSeparator) { roisFile => 
            assertThrows[DecoderError]{ unsafeRead[NucleusLabeledProximityAssessedRoi](roisFile) }
        }
    }

    test("NucleusLabeledProximityAssessedRoi cannot be parsed when either proximal ROIs column is missing.") {
        val datas = List(
            """
            index,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,nucleusNumber,tooCloseRois
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,2,
            """.toLines,
            """
            index,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,nucleusNumber,mergeRois
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,2,
            """.toLines,
            """
            index,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,nucleusNumber
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,2
            """.toLines,
        )

        // Additional component to CsvRowDecoder[DetectedSpotRoi, HeaderField] to derive 
        // CsvRowDecoder[NucleusLabeledProximityAssessedRoi, HeaderField]
        given CsvRowDecoder[NuclearDesignation, HeaderField] = getCsvRowDecoderForNuclearDesignation()
        
        forAll (Table(("initData"), datas*)) { initData => 
            withTempFile(initData, Delimiter.CommaSeparator) { roisFile => 
                assertThrows[DecoderError]{ unsafeRead[NucleusLabeledProximityAssessedRoi](roisFile) }
            }
        }
    }

    private def unsafeRead[A](roisFile: os.Path)(using CsvRowDecoder[A, String], CharLikeChunks[IO, Byte]): List[A] = 
        readCsvToCaseClasses[A](roisFile).unsafeRunSync()

    extension (example: String)
        // Utility function to trim line endings and whitespace, accounting for formatting of raw example data.
        private def toLines: List[String] = example.split("\n").map(_.trim).filterNot(_.isEmpty).toList.map(_ ++ "\n")
end TestParseRoisCsv
