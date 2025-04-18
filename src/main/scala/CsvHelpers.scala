package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import cats.syntax.all.*
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

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

    extension (maybe: Try[String])
      private def continueParse[A](lift: String => Either[String, A]) = (maybe.toEither.leftMap(_.getMessage) >>= lift).toValidatedNel
end CsvHelpers
