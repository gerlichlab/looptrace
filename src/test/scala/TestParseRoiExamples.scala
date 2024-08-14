package at.ac.oeaw.imba.gerlich.looptrace
package csv

import cats.effect.IO
import cats.effect.unsafe.implicits.global // needed to call unsafeRunSync()
import fs2.{ Pipe, Stream }
import fs2.data.csv.*

import org.scalacheck.*
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import io.github.iltotore.iron.{ autoCastIron, autoRefine }

import at.ac.oeaw.imba.gerlich.gerlib.geometry.BoundingBox
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid

import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{
    ColumnName, 
    ColumnNameLike, 
    readCsvToCaseClasses, 
    writeCaseClassesToCsv, 
}
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.roi.getCsvRowDecoderForDetectedSpot
import at.ac.oeaw.imba.gerlich.gerlib.roi.DetectedSpot
import at.ac.oeaw.imba.gerlich.gerlib.roi.measurement.{ Area, MeanIntensity }
import at.ac.oeaw.imba.gerlich.gerlib.testing.catsScalacheck.given
import at.ac.oeaw.imba.gerlich.gerlib.testing.csv.given

import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.space.{BoundingBox as BB, *}

/**
  * Tests for parsing different types of ROIs.
  */
class TestParseRoiExamples extends AnyFunSuite, GenericSuite, should.Matchers, ScalaCheckPropertyChecks:

    type HeaderField = String
    type RawCoordinate = Double

    test("Small detected spot ROI example parses correctly.") {
        import fs2.data.text.utf8.byteStreamCharLike // to prove CharLikeChunks[cats.effect.IO, Byte]

        withTempFile(Data.linesToWrite, Delimiter.CommaSeparator){ roisFile =>
            val observed: List[DetectedSpotRoi] = readCsvToCaseClasses[DetectedSpotRoi](roisFile).unsafeRunSync()
            observed.length shouldEqual Data.expectedRecords.length // quick, simplifying check
            observed shouldEqual Data.expectedRecords // full check
        }
    }
    
    test("Small nucleus label attempted spot ROI example parses correctly.") { pending }

    /** Abbreviation for the heavily used column name like data type */
    private type CN[A] = ColumnNameLike[A]

    /** The columns which correspond to the fields of a detected spoT */
    private type SpotColumns = (
        CN[FieldOfViewLike], 
        CN[ImagingTimepoint], 
        CN[ImagingChannel], 
        CN[ZCoordinate], 
        CN[YCoordinate], 
        CN[XCoordinate], 
        CN[Area], 
        CN[MeanIntensity],
    )

    private type BoxColumns = (
        CN[ZCoordinate], 
        CN[ZCoordinate], 
        CN[YCoordinate],
        CN[YCoordinate], 
        CN[XCoordinate], 
        CN[XCoordinate], 
    )

    // test("DetectedSpotRoi records roundtrip through CSV.") {
    //     // Choose an arbitrary permutation of the fixed input ROI records.
    //     given Arbitrary[List[DetectedSpotRoi]] = Arbitrary{ Gen.oneOf(Data.expectedRecords.permutations.toList) }
        
    //     def genRois: Gen[List[DetectedSpotRoi]] = Arbitrary.arbitrary[List[DetectedSpotRoi]]

    //     // Used to determine tempfile extension and the character(s) to avoid in generated column names
    //     val delimiter = Delimiter.CommaSeparator

    //     // Generate the column names such that there are no repeats and such that no column name contains the delimiter.
    //     def genCols: Gen[SpotColumns] = Arbitrary.arbitrary[SpotColumns].suchThat{
    //         case (fov, time, channel, z, y, x, area, intensity) => 
    //             val rawNames = List(fov, time, channel, z, y, x, area, intensity).map(_.value)
    //             rawNames.forall{ cn => !cn.contains(delimiter.sep) } && rawNames.length === rawNames.toSet.size
    //     }
    
    //     forAll (genRois, genCols) { (inputRecords: List[DetectedSpotRoi], colnames: SpotColumns) => 

    //         withTempFile(delimiter){ roisFile =>
    //             /* First, write the records to CSV */
    //             val sink: Pipe[IO, DetectedSpotRoi, Nothing] = writeCaseClassesToCsv[DetectedSpotRoi](roisFile)
    //             Stream.emits(inputRecords).through(sink).compile.drain.unsafeRunSync()

    //             /* Then, do the parse-and-check. */
    //             val outputRecords: List[DetectedSpotRoi] = readCsvToCaseClasses[DetectedSpotRoi](roisFile).unsafeRunSync()
    //             outputRecords shouldEqual inputRecords
    //         }
    //     }
    // }

    object Data:
        private val inputLines = 
            """
            ,fieldOfView,timepoint,roiChannel,zc,yc,xc,area,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax,nucleusNumber
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,0
            1,P0001.zarr,80,2,17.994259347453493,24.042015416774795,1360.0069098862991,213.58943029032,117.1394688491732,11.994259347453491,23.994259347453493,12.042015416774795,36.0420154167748,1348.0069098862991,1372.0069098862991,0
            """.cleanLines
        val linesToWrite = inputLines.map(_ ++ "\n")

        val pos = PositionName("P0001.zarr")
        
        private val expectedRecord1: DetectedSpotRoi = {
            val xBounds = XCoordinate(859.9833511648726) -> XCoordinate(883.9833511648726)
            val yBounds = YCoordinate(219.9874778925304) -> YCoordinate(243.9874778925304)
            val zBounds = ZCoordinate(-2.092371012467521) -> ZCoordinate(9.90762898753248)
            DetectedSpotRoi(
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
                BoundingBox(
                    BoundingBox.Interval.unsafeFromTuple(xBounds), 
                    BoundingBox.Interval.unsafeFromTuple(yBounds), 
                    BoundingBox.Interval.unsafeFromTuple(zBounds),
                ),
            )
        }

        private val expectedRecord2: DetectedSpotRoi = {
            val xBounds = XCoordinate(1348.0069098862991) -> XCoordinate(1372.0069098862991)
            val yBounds = YCoordinate(12.042015416774795) -> YCoordinate(36.0420154167748)
            val zBounds = ZCoordinate(11.994259347453491) -> ZCoordinate(23.994259347453493)
            DetectedSpotRoi(
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
                BoundingBox(
                    BoundingBox.Interval.unsafeFromTuple(xBounds), 
                    BoundingBox.Interval.unsafeFromTuple(yBounds), 
                    BoundingBox.Interval.unsafeFromTuple(zBounds),
                ),
            )
        }

        def expectedRecords: List[DetectedSpotRoi] = List(expectedRecord1, expectedRecord2)
    end Data

    extension (example: String)
        // Utility function to trim line endings and whitespace, accounting for formatting of raw example data.
        private def cleanLines: List[String] = example.split("\n").map(_.trim).filterNot(_.isEmpty).toList
end TestParseRoiExamples
