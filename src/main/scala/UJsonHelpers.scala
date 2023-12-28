package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import cats.data.{ NonEmptyList as NEL, ValidatedNel }
import cats.syntax.all.*

/** Helpers for working with the excellent uJson project */
object UJsonHelpers:
    /** Lift floating-point to JSON number, through floating-point. */
    given liftDouble: (Double => ujson.Num) = ujson.Num.apply

    /** Lift integer to JSON number, through floating-point. */
    given liftInt: (Int => ujson.Num) = z => liftDouble(z.toDouble)
    
    /** Lift floating-point to JSON number. */
    given liftStr: (String => ujson.Str) = ujson.Str.apply

    /**
      * Try to parse an {@code A} value from the value at the given key, using {@code Int} as intermediary.
      *
      * @param key The key in a JSON object map at which to fetch a raw value, to be read initially as {@code Int}
      * @param lift How to make an {@code A} from an {@code Int}
      * @return A function accepting a JSON value (assumed to be an object) and trying the parse at the given key
      */
    def fromJsonThruInt[A](key: String, lift: Int => Either[String, A]) = 
        (json: ujson.Value) => (Try{ json(key).int }.toEither.leftMap(_.getMessage) >>= lift).toValidatedNel

    /** Lift mapping with text keys to JSON object. */
    def liftMap[V](using conv: V => ujson.Value): (Map[String, V] => ujson.Obj) = m => ujson.Obj.from(m.view.mapValues(conv).toList)

    /** Read given JSON file into value of target type. */
    def readJsonFile[A](jsonFile: os.Path)(using upickle.default.Reader[A]): A = upickle.default.read[A](os.read(jsonFile))

    /**
      * Try to extract an {@code A} value from JSON object, at given {@code key}.
      *
      * @tparam A Type of value to try to extract
      * @param key The key at which to try to extract a value
      * @param lift How to convert text into a value of {@code A}
      * @param json The object from which to extract a value at given key
      * @return Either a collection of error messages or the extracted value
      */
    def safeExtract[A](key: String, lift: String => A)(json: ujson.Value): ValidatedNel[String, A] = 
        safeExtractStr(key)(json) `map` lift
    
    def safeExtractE[A](key: String, lift: String => Either[String, A])(json: ujson.Value): ValidatedNel[String, A] = 
        (safeExtractStr(key)(json).toEither >>= lift.fmap(_.leftMap(NEL.one))).toValidated

    /** Try to extract a text value at the given key in the given (presumed Object) JSON value. */
    def safeExtractStr(key: String)(json: ujson.Value): ValidatedNel[String, String] = 
        Try{ json(key).str }.toEither.leftMap(_.getMessage).toValidatedNel

end UJsonHelpers
