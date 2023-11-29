package at.ac.oeaw.imba.gerlich.looptrace

import java.nio.file.FileAlreadyExistsException
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

    /** Exception for when at least one row has field names not matching those of a header */
    private[looptrace] final case class FieldNameColumnNameMismatchException(records: NEL[ErrorData]) extends Throwable

    /** Order each row's fields according to the header, and ensure the exact match between fields and columns. */
    def prepCsvWrite(header: List[String])(rows: Iterable[CsvRow]): Either[IllegalArgumentException | FieldNameColumnNameMismatchException, List[List[String]]] = for {
        _ <- (header.toSet.size === header.length).either(IllegalArgumentException(s"Repeat elements present in header: $header"), ())
        getKey = header.zipWithIndex.toMap.apply
        sortRowFields = (row: CsvRow) => {
            val ordered = row.toList.sortBy((k, _) => getKey(k))
            (ordered.map(_._1) === header).either("Sorted field names differ from header", ordered.map(_._2))
        }
        (errors, records) = Alternative[List].separate(
            NonnegativeInt.indexed(rows.toList).map{ case (row, idx) => sortRowFields(row).leftMap((idx, _, row)) })
        result <- errors.toNel.toLeft(records).leftMap(FieldNameColumnNameMismatchException.apply)
    } yield result
    
    /** Safely write rows to CSV with header, ensuring each has exactly the header's fields, and is ordered accordingly. */
    def writeAllCsv(f: os.Path, header: List[String], rows: Iterable[CsvRow], handleExtant: ExtantOutputHandler): Boolean = {
        val maybeWrite: Either[Throwable, os.Source => Boolean] = handleExtant.getSimpleWriter(f)
        val maybeLines: Either[Throwable, os.Source] = prepCsvWrite(header)(rows).map(recs => (header :: recs).map(_.mkString(",") ++ "\n"))
        (maybeWrite <*> maybeLines).fold(throw _, identity)
    }

end CsvHelpers
