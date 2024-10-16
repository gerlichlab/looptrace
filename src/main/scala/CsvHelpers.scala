package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import cats.Alternative
import cats.data.{ NonEmptyList as NEL }
import cats.syntax.all.*
import mouse.boolean.*
import com.github.tototoshi.csv.*

import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

/** Helpers for working with CSV files and data */
object CsvHelpers:

    /** Alias for a bundle of line number, error message, and single row/record fields */
    type ErrorData = (LineNumber, ErrMsg, Map[String, String])
    
    /** Alias for when text represents an error message */
    type ErrMsg = String
    
    /** Alias for when a nonnegative integer represents a line number */
    type LineNumber = NonnegativeInt

    /** Exception for when at least one row has field names not matching those of a header */
    private[looptrace] final case class FieldNameColumnNameMismatchException(records: NEL[ErrorData]) extends Throwable

    /** Order each row's fields according to the header, and ensure the exact match between fields and columns. */
    def prepCsvWrite(header: List[String])(rows: Iterable[Map[String, String]]): Either[IllegalArgumentException | FieldNameColumnNameMismatchException, List[List[String]]] = for {
        _ <- (header.toSet.size === header.length).either(IllegalArgumentException(s"Repeat elements present in header: $header"), ())
        getKey = header.zipWithIndex.toMap.apply
        sortRowFields = (row: Map[String, String]) => {
            val ordered = row.toList.sortBy((k, _) => getKey(k))
            (ordered.map(_._1) === header).either("Sorted field names differ from header", ordered.map(_._2))
        }
        (errors, records) = Alternative[List].separate(
            NonnegativeInt.indexed(rows.toList).map{ case (row, idx) => sortRowFields(row).leftMap((idx, _, row)) })
        result <- errors.toNel.toLeft(records).leftMap(FieldNameColumnNameMismatchException.apply)
    } yield result

    extension (maybe: Try[String])
        private def continueParse[A](lift: String => Either[String, A]) = (maybe.toEither.leftMap(_.getMessage) >>= lift).toValidatedNel

    /** Try to extract a value from a particular index in a sequence, and then read it into a value of the target type. */
    def safeGetFromRow[A](idx: Int, lift: String => Either[String, A])(row: List[String]) = Try(row(idx)).continueParse(lift)

    /** Try to extract a value from a particular index in a sequence, and then read it into a value of the target type. */
    def safeGetFromRow[A](idx: Int, lift: String => Either[String, A])(row: Array[String]) = Try(row(idx)).continueParse(lift)

    /**
      * Try reading an {@code A} value from the given key in the key-value representation of a CSV row.
      *
      * @tparam A The type of value to attempt to parse
      * @param key The key of the field to read/parse
      * @param lift How to parse an {@code A} from raw string
      * @param row The row from which to parse the value
      * @return Either a [[scala.util.Left]]-wrapped error message, or a [[scala.util.Right]]-wrapped parsed value
      */
    def safeGetFromRow[A](key: String, lift: String => Either[String, A])(row: Map[String, String]) = 
      row.get(key)
        .toRight(s"Row lacks key '$key'")
        .flatMap{ raw => lift(raw).leftMap(msg => s"Failed to parse value ($raw) from key '$key': $msg") }
        .toValidatedNel

    /**
      * Read given file as CSV with header, and handle resource safety.
      *
      * @param f The path to the file to read as CSV
      * @return Either a [[scala.util.Left]]-wrapped exception or a [[scala.util.Right]]-wrapped pair of columns and list of row records
      */
    def safeReadAllWithOrderedHeaders(f: os.Path): Either[Throwable, (List[String], List[Map[String, String]])] = for {
        reader <- Try{ CSVReader.open(f.toIO) }.toEither
        result <- Try{ reader.allWithOrderedHeaders() }.toEither
        _ = reader.close()
    } yield result

    /**
      * Try writing a CSV file of the given header and rows data, using the given function already parameterised with target file.
      *
      * @param writeLines How to write data to a fixed (already parameterised) file, possibly skipping the write e.g. if target already exists
      * @param header The header fields to write to the target file
      * @param rows The individual rows/records of data to write as CSV
      * @return Either a [[scala.util.Left]]-wrapped exception, or a [[scala.util.Right]]-wrapped flag indicating whether the file was written
      */
    def writeAllCsvSafe(writeLines: os.Source => Boolean)(header: List[String], rows: Iterable[Map[String, String]]): Either[IllegalArgumentException | FieldNameColumnNameMismatchException, Boolean] = 
        prepCsvWrite(header)(rows).map(recs => (header :: recs).map(_.mkString(",") ++ "\n")).map(recs => writeLines(recs))

    /**
      * Write given header and rows data as CSV to a target parameterised already in the given writing function
      *
      * @param writeLines How to write data to a fixed (already parameterised) file, possibly skipping the write e.g. if target already exists
      * @param header The columns / header fields to write
      * @param rows The individual rows / records to write as CSV
      * @return A flag indicating if the file was written (most likely), determined by the given writing function
      */
    def writeAllCsvUnsafe(writeLines: os.Source => Boolean)(header: List[String], rows: Iterable[Map[String, String]]): Boolean = 
        writeAllCsvSafe(writeLines)(header, rows).fold(throw _, identity)

end CsvHelpers
