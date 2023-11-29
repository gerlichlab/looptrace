package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import cats.Alternative
import cats.data.{ NonEmptyList as NEL }
import cats.syntax.all.*
import mouse.boolean.*

/** Helpers for working with CSV files and data */
object CsvHelpers:
    type ErrorData = (LineNumber, ErrMsg, CsvRow)
    type ErrMsg = String
    type LineNumber = NonnegativeInt

    /** Order each row's fields according to the header, and ensure the exact match between fields and columns. */
    private def prepCsvWrite(header: List[String])(rows: Iterable[CsvRow]): Either[NEL[ErrorData], List[List[String]]] = {
        require(header.toSet.size === header.length, s"Repeat elements present in header: $header")
        def getKey = header.zipWithIndex.toMap.apply
        def sortRowFields = (row: CsvRow) => {
            val ordered = row.toList.sortBy((k, _) => getKey(k))
            (ordered.map(_._1) === header).either("Sorted field names differ from header", ordered.map(_._2))
        }
        val (errors, records) = Alternative[List].separate(
            NonnegativeInt.indexed(rows.toList).map( (row, idx) => sortRowFields(row).leftMap(err => (idx, err, row)) ))
        errors.toNel.toLeft(records)
    }
    
    /** Safely write rows to CSV with header, ensuring each has exactly the header's fields, and is ordered accordingly. */
    def writeAllCsv(f: os.Path, header: List[String], rows: Iterable[CsvRow]): Either[NEL[ErrorData], os.Path] = {
        prepCsvWrite(header)(rows).map{ records => 
            val lines = (header :: records).map(_.mkString(",") ++ "\n")
            os.write(f, lines)
            f
        }
    }

end CsvHelpers
