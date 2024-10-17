package at.ac.oeaw.imba.gerlich.looptrace.csv
package instances

import fs2.data.csv.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.getCsvRowDecoderForSingleton
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.{ FieldOfViewColumnName, TimepointColumnName }
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given

/** CSV-related typeclass instances for imaging-related data types */
trait ImagingCsvInstances:
    given CsvRowDecoder[FieldOfViewLike, String] = 
        getCsvRowDecoderForSingleton(FieldOfViewColumnName)
    
    given CsvRowDecoder[ImagingTimepoint, String] = 
        getCsvRowDecoderForSingleton(TimepointColumnName)
end ImagingCsvInstances
