package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import cats.syntax.all.*
import com.github.tototoshi.csv.*

/** Helpers for working with CSV files and data */
object CsvHelpers:
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

    extension (maybe: Try[String])
      private def continueParse[A](lift: String => Either[String, A]) = (maybe.toEither.leftMap(_.getMessage) >>= lift).toValidatedNel
end CsvHelpers
