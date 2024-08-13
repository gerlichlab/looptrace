package at.ac.oeaw.imba.gerlich.looptrace.csv
package instances

import fs2.data.csv.CsvRowDecoder

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.getCsvRowDecoderForProduct2
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.roi.DetectedSpot

import at.ac.oeaw.imba.gerlich.looptrace.DetectedSpotRoi
import at.ac.oeaw.imba.gerlich.looptrace.NucleusLabelAttemptedRoi
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

/** Typeclass instances related to CSV, for ROI-related data types */
trait RoiCsvInstances:
    private type Header = String
    
    given csvRowDecoderForDetectedSpotRoi(using 
        CsvRowDecoder[DetectedSpot[Double], Header],
        CsvRowDecoder[BoundingBox, Header],
    ): CsvRowDecoder[DetectedSpotRoi, Header] = 
        getCsvRowDecoderForProduct2(DetectedSpotRoi.apply)

    given csvRowDecoderForNucleusLabelAttemptedRoi(using 
        CsvRowDecoder[DetectedSpotRoi, Header],
        CsvRowDecoder[NuclearDesignation, Header]
    ): CsvRowDecoder[NucleusLabelAttemptedRoi, Header] = 
        getCsvRowDecoderForProduct2(NucleusLabelAttemptedRoi.apply)
end RoiCsvInstances
