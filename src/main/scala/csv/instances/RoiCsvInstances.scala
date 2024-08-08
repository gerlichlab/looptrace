package at.ac.oeaw.imba.gerlich.looptrace.csv
package instances

import fs2.data.csv.CsvRowDecoder

import at.ac.oeaw.imba.gerlich.gerlib.io.csv.getCsvRowDecoderForProduct2
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given

import at.ac.oeaw.imba.gerlich.looptrace.DetectedSpotRoi
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.spatial.given
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

/** Typeclass instances related to CSV, for ROI-related data types */
trait RoiCsvInstances:
    given csvRowDecoderForDetectedSpotRoi(using CsvRowDecoder[BoundingBox, String]): CsvRowDecoder[DetectedSpotRoi, String] = 
        getCsvRowDecoderForProduct2(DetectedSpotRoi.apply)
end RoiCsvInstances
