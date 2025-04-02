package at.ac.oeaw.imba.gerlich.looptrace.csv
package instances

import cats.syntax.all.*
import fs2.data.csv.*
import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.getCsvRowDecoderForSingleton
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.{ FieldOfViewColumnName, TimepointColumnName }
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.OneBasedFourDigitPositionName

/** CSV-related typeclass instances for imaging-related data types */
trait ImagingCsvInstances:
    given CellDecoder[OneBasedFourDigitPositionName] = new:
        override def apply(cell: String): DecoderResult[OneBasedFourDigitPositionName] = 
            OneBasedFourDigitPositionName
                .fromString(true)(cell)
                .leftMap(DecoderError(_, None))

    given (enc: SimpleShow[OneBasedFourDigitPositionName]) => CellEncoder[OneBasedFourDigitPositionName] = new:
        override def apply(cell: OneBasedFourDigitPositionName): String = 
            import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*
            cell.show_

    given CsvRowDecoder[FieldOfViewLike, String] = 
        getCsvRowDecoderForSingleton(FieldOfViewColumnName)
    
    given CsvRowDecoder[ImagingTimepoint, String] = 
        getCsvRowDecoderForSingleton(TimepointColumnName)
end ImagingCsvInstances
