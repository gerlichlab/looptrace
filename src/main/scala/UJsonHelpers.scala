package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import cats.data.{NonEmptyList, NonEmptySet, ValidatedNel}
import cats.syntax.all.*
import mouse.boolean.*
import upickle.core.Visitor
import upickle.default.*

import at.ac.oeaw.imba.gerlich.looptrace.collections.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Helpers for working with the excellent uJson project
  *
  * @author
  *   Vince Reuter
  */
object UJsonHelpers:
  /** Lift floating-point to JSON number, through floating-point. */
  given liftDouble: (Double => ujson.Num) = ujson.Num.apply

  /** Lift integer to JSON number, through floating-point. */
  given liftInt: (Int => ujson.Num) = z => liftDouble(z.toDouble)

  /** Lift floating-point to JSON number. */
  given liftStr: (String => ujson.Str) = ujson.Str.apply

  /** Entity which admits a {@code String}-keyed mapping to {@code ujson.Value}
    */
  trait JsonMappable[A]:
    /** Define an instance by saying how to make a string-keyed mapping from an
      * {@code A}
      */
    def toJsonMap: A => JsonMappable.JMap
  end JsonMappable

  object UPickleCatsInstances:
    import cats.*
    import upickle.default.*
    given [T] => Contravariant[Writer]:
      override def contramap[A, B](wa: Writer[A])(f: B => A): Writer[B] =
        new Writer[B] {
          override def write0[V](out: Visitor[?, V], v: B): V =
            wa.write0(out, f(v))
        }
  end UPickleCatsInstances

  object JsonMappable:
    type JMap = Map[String, ujson.Value]
    object JMap:
      def empty: JMap = Map()

    /** A JMap is already itself a JMap, so just return itself. */
    given jsonMappableForJsonMap: JsonMappable[JMap]:
      override def toJsonMap = identity

    extension [A](a: A)(using ev: JsonMappable[A])
      def toJsonMap: JMap = ev.toJsonMap(a)
      def toJsonObject: ujson.Obj = this.toJsonObject(a.toJsonMap)

    /** Combine 2 maps from text key to JSON value, succeeding iff no keyset
      * overlap, otherwise collecting overlapping keys.
      */
    def combineSafely = (m1: JMap, m2: JMap) => {
      val result = m1 ++ m2
      (result.size === m1.size + m2.size)
        .either((m1.keySet & m2.keySet).toNonEmptySetUnsafe, result)
    }

    /** Combine arbitrarily numerous maps from text key to JSON value,
      * succeeding iff no keyset overlap, otherwise collecting overlapping keys.
      */
    def combineSafely(ms: List[JMap]): Either[NonEmptySet[String], JMap] =
      ms.foldRight(JMap.empty.asRight[NonEmptySet[String]]) { case (m, acc) =>
        acc.leftMap(_ ++ m.keySet).flatMap(combineSafely(m, _))
      }

    /** Create an instance with the given function as the
      * characteristic/defining function.
      */
    def instance[A]: (A => JsonMappable.JMap) => JsonMappable[A] =
      f => new JsonMappable[A] { override def toJsonMap: A => JMap = f }

    private final def toJsonObject(m: JMap): ujson.Obj = m.toList.match {
      case kv :: tail => ujson.Obj(kv, tail*)
      case Nil        => ujson.Obj()
    }
  end JsonMappable

  /** Error type for when key sets overlap but shouldn't */
  class RepeatedKeysException[A](keys: NonEmptySet[A]) extends Throwable

  /** Try to parse an {@code A} value from the value at the given key, using
    * {@code Int} as intermediary.
    *
    * @param key
    *   The key in a JSON object map at which to fetch a raw value, to be read
    *   initially as {@code Int}
    * @param lift
    *   How to make an {@code A} from an {@code Int}
    * @return
    *   A function accepting a JSON value (assumed to be an object) and trying
    *   the parse at the given key
    */
  def fromJsonThruInt[A](key: String, lift: Int => Either[String, A]) =
    (json: ujson.Value) =>
      (Try { json(key).int }.toEither
        .leftMap(_.getMessage) >>= lift).toValidatedNel

  /** Lift mapping with text keys to JSON object. */
  def liftMap[V](using conv: V => ujson.Value): (Map[String, V] => ujson.Obj) =
    m => ujson.Obj.from(m.view.mapValues(conv).toList)

  /** Read given JSON file into value of target type. */
  def readJsonFile[A](jsonFile: os.Path)(using upickle.default.Reader[A]): A =
    upickle.default.read[A](os.read(jsonFile))

  /** Try to extract an {@code A} value from JSON object, at given {@code key}.
    *
    * @tparam A
    *   Type of value to try to extract
    * @param key
    *   The key at which to try to extract a value
    * @param lift
    *   How to convert text into a value of {@code A}
    * @param json
    *   The object from which to extract a value at given key
    * @return
    *   Either a collection of error messages or the extracted value
    */
  def safeExtract[A](key: String, lift: String => A)(
      json: ujson.Value
  ): ValidatedNel[String, A] =
    safeExtractStr(key)(json) `map` lift

  def safeExtractE[A](key: String, lift: String => Either[String, A])(
      json: ujson.Value
  ): ValidatedNel[String, A] =
    (safeExtractStr(key)(json).toEither >>= lift.fmap(
      _.leftMap(NonEmptyList.one)
    )).toValidated

  /** Try to extract a text value at the given key in the given (presumed
    * Object) JSON value.
    */
  def safeExtractStr(key: String)(
      json: ujson.Value
  ): ValidatedNel[String, String] =
    Try { json(key).str }.toEither.leftMap(_.getMessage).toValidatedNel

  /** Try to read the given value to an `A`, otherwise give an error message. */
  def safeReadAs[A: Reader](json: ujson.Readable): Either[String, A] =
    Try { read[A](json) }.toEither.leftMap(_.getMessage)

end UJsonHelpers
