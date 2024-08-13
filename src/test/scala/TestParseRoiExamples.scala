package at.ac.oeaw.imba.gerlich.looptrace
package csv

import cats.effect.unsafe.implicits.global // needed to call unsafeRunSync()
import fs2.data.csv.*

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should

import io.github.iltotore.iron.{ autoCastIron, autoRefine }

import at.ac.oeaw.imba.gerlich.gerlib.geometry.BoundingBox
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid

import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{
    ColumnName, 
    ColumnNameLike, 
    readCsvToCaseClasses, 
}
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.roi.{
    getCsvRowDecoderForDetectedSpot,
    oldCsvRowDecoderForBoundingBox, 
}
import at.ac.oeaw.imba.gerlich.gerlib.roi.DetectedSpot
import at.ac.oeaw.imba.gerlich.gerlib.roi.measurement.{ Area, MeanIntensity }

import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.space.{BoundingBox as BB, *}

/**
  * Tests for parsing different types of ROIs.
  */
class TestParseRoiExamples extends AnyFunSuite, GenericSuite, should.Matchers:

    type HeaderField = String
    type RawCoordinate = Double

    test("Small detected spot ROI example parses correctly.") {
        import fs2.data.text.utf8.byteStreamCharLike // to prove CharLikeChunks[cats.effect.IO, Byte]
        
        val expected: List[DetectedSpotRoi] = List(Data.expectedRecord1, Data.expectedRecord2)
        
        /* Decoders for the components of the target case class */
        given CsvRowDecoder[DetectedSpot[RawCoordinate], HeaderField] = getCsvRowDecoderForDetectedSpot(
            fovCol = ColumnName[FieldOfViewLike]("position"), 
            timeCol = ColumnName[ImagingTimepoint]("frame"), 
            channelCol = ColumnName[ImagingChannel]("ch"), 
        )
        given CsvRowDecoder[BoundingBox[RawCoordinate], HeaderField] = oldCsvRowDecoderForBoundingBox[RawCoordinate]
        
        withTempFile(Data.linesToWrite, Delimiter.CommaSeparator){ roisFile =>
            val observed: List[DetectedSpotRoi] = readCsvToCaseClasses[DetectedSpotRoi](roisFile).unsafeRunSync()
            observed.length shouldEqual expected.length // quick, simplifying check
            observed shouldEqual expected // full check
        }
    }
    
    test("Small nucleus label attempted spot ROI example parses correctly.") { pending }

    object Data:
        private val inputLines = 
            """
            ,position,frame,ch,zc,yc,xc,area,intensity_mean,z_min,z_max,y_min,y_max,x_min,x_max,nuc_label
            0,P0001.zarr,79,0,3.907628987532479,231.9874778925304,871.9833511648726,240.00390423,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726,0
            1,P0001.zarr,80,2,17.994259347453493,24.042015416774795,1360.0069098862991,213.58943029032,117.1394688491732,11.994259347453491,23.994259347453493,12.042015416774795,36.0420154167748,1348.0069098862991,1372.0069098862991,0
            """.cleanLines
        val linesToWrite = inputLines.map(_ ++ "\n")

        val pos = PositionName("P0001.zarr")
        
        val expectedRecord1: DetectedSpotRoi = {
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

        val expectedRecord2: DetectedSpotRoi = {
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

    end Data

    extension (example: String)
        // Utility function to trim line endings and whitespace, accounting for formatting of raw example data.
        private def cleanLines: List[String] = example.split("\n").map(_.trim).filterNot(_.isEmpty).toList
end TestParseRoiExamples
