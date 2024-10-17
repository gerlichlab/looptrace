package at.ac.oeaw.imba.gerlich.looptrace

import fs2.data.csv.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNameLike

/** IO functionality specific to CSV */
package object csv:
    def getCsvRowDecoderForImagingChannel(column: ColumnNameLike[ImagingChannel])(using CellDecoder[ImagingChannel]): CsvRowDecoder[ImagingChannel, String] = new:
        override def apply(row: RowF[Some, String]): DecoderResult[ImagingChannel] = row.as(column.value)
