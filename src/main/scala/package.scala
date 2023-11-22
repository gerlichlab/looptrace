package at.ac.oeaw.imba.gerlich

import java.io.File
import scala.util.Try
import upickle.default.*
import cats.{ Eq, Order, Show }
import cats.data.{ NonEmptyList as NEL, ValidatedNel }
import cats.syntax.contravariant.*
import cats.syntax.either.*
import cats.syntax.eq.*
import cats.syntax.functor.*
import cats.syntax.option.*
import cats.syntax.show.*

import mouse.boolean.*
import scopt.Read
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.readJsonFile

/** Chromatin fiber tracing with FISH probes */
package object looptrace {
    val VersionName = "0.1.0-SNAPSHOT"

    type ErrorMessages = NEL[String]
    type ErrMsgsOr[A] = Either[ErrorMessages, A]

    def writeTextFile(target: os.Path, data: Iterable[Array[String]], delimiter: Delimiter) = 
        os.write(target, data.map(delimiter.join.andThen(_ ++ "\n")))

    extension (ps: Iterable[Boolean])
        def all: Boolean = ps.forall(identity)

    extension (p: os.Path)
        def parent: os.Path = p / os.up

    extension (v: ujson.Value)
        def int: Int = {
            val x = v.num
            val z = x.toInt
            if (x === z) then z else throw new ujson.Value.InvalidData(v, "Cannot convert to integer")
        }

    extension [A](arr: Array[A])
        def lookup(a: A): Option[Int] = arr.indexOf(a) match {
            case -1 => None
            case i => i.some
        }

    extension [A](t: Try[A])
        def toValidatedNel: ValidatedNel[Throwable, A] = t.toEither.toValidatedNel

    /** Find a field in a header and use the index to build a row record parser for that field. */
    def buildFieldParse[A](name: String, parse: String => Either[String, A])(header: Array[String]): ValidatedNel[String, Array[String] => ValidatedNel[String, A]] = {
        header.lookup(name).toRight(f"Missing field in header: $name").map{ i => {
            (rr: Array[String]) => Try(rr(i))
                .toEither
                .leftMap(_ => s"Out of bounds finding value for '$name' in record with ${rr.length} fields: $i")
                .flatMap(parse)
                .toValidatedNel
        } }.toValidatedNel
    }

    /** Get the labels of a Product. */
    inline def labelsOf[A](using p: scala.deriving.Mirror.ProductOf[A]) = scala.compiletime.constValueTuple[p.MirroredElemLabels]

    /** Try to parse the given string as a decimal value. */
    def safeParseDouble(s: String): Either[String, Double] = Try{ s.toDouble }.toEither.leftMap(_.getMessage)
    
    /** Try to parse a positive real number from the given string. */
    def safeParsePosNum = safeParseDouble.fmap(_.flatMap(PositiveReal.either))

    /** Try to parse the given string as an integer. */
    def safeParseInt(s: String): Either[String, Int] = Try{ s.toInt }.toEither.leftMap(_.getMessage)

    /** Allow custom types as CLI parameters. */
    object ScoptCliReaders:
        given pathRead(using fileRead: Read[File]): Read[os.Path] = fileRead.map(os.Path.apply)
        given nonNegIntRead(using intRead: Read[Int]): Read[NonnegativeInt] = intRead.map(NonnegativeInt.unsafe)
        given nonNegRealRead(using numRead: Read[Double]): Read[NonnegativeReal] = numRead.map(NonnegativeReal.unsafe)
        given posIntRead(using intRead: Read[Int]): Read[PositiveInt] = intRead.map(PositiveInt.unsafe)
        given posRealRead(using numRead: Read[Double]): Read[PositiveReal] = numRead.map(PositiveReal.unsafe)
    end ScoptCliReaders

    /** Refinement type for nonnegative integers */
    opaque type NonnegativeInt <: Int = Int
    
    /** Helpers for working with nonnegative integers */
    object NonnegativeInt:
        inline def apply(z: Int): NonnegativeInt = 
            inline if z < 0 then compiletime.error("Negative integer where nonnegative is required!")
            else (z: NonnegativeInt)
        def either(z: Int): Either[String, NonnegativeInt] = maybe(z).toRight(s"Cannot refine as nonnegative: $z")
        def indexed[A](xs: List[A]): List[(A, NonnegativeInt)] = {
            // guaranteed nonnegative by construction here
            xs.zipWithIndex.map{ case (x, i) => x -> unsafe(i) }
        }
        def maybe(z: Int): Option[NonnegativeInt] = (z >= 0).option{ (z: NonnegativeInt) }
        def unsafe(z: Int): NonnegativeInt = either(z).fold(msg => throw new NumberFormatException(msg), identity)
        given nonnegativeIntOrder(using intOrd: Order[Int]): Order[NonnegativeInt] = intOrd.contramap(identity)
    end NonnegativeInt

    /** Refinement type for nonnegative integers */
    opaque type PositiveInt <: Int = Int
    
    /** Helpers for working with nonnegative integers */
    object PositiveInt:
        inline def apply(z: Int): PositiveInt = 
            inline if z <= 0 then compiletime.error("Non-positive integer where positive is required!")
            else (z: PositiveInt)
        def either(z: Int): Either[String, PositiveInt] = maybe(z).toRight(s"Cannot refine as positive: $z")
        def maybe(z: Int): Option[PositiveInt] = (z > 0).option{ (z: PositiveInt) }
        def unsafe(z: Int): PositiveInt = either(z).fold(msg => throw new NumberFormatException(msg), identity)
        given posIntOrder(using intOrd: Order[Int]): Order[PositiveInt] = intOrd.contramap(identity)
        given posIntRW(using intRW: ReadWriter[Int]): ReadWriter[PositiveInt] = intRW.bimap(identity, _.int)
        extension (n: PositiveInt)
            def asNonnegative: NonnegativeInt = NonnegativeInt.unsafe(n)
    end PositiveInt

    /** Refinement for positive real values */
    opaque type PositiveReal <: Double = Double

    /** Helpers for working with positive real numbers */
    object PositiveReal:
        inline def apply(x: Double): PositiveReal = 
            inline if x > 0 then (x: PositiveReal)
            else compiletime.error("Non-positive value where positive is required!")
        def either(x: Double): Either[String, PositiveReal] = maybe(x).toRight(s"Cannot refine as positive: $x")
        def maybe(x: Double): Option[PositiveReal] = (x > 0).option{ (x: PositiveReal) }
        def unsafe(x: Double): PositiveReal = either(x).fold(msg => throw new NumberFormatException(msg), identity)
        given posRealOrd(using numOrd: Order[Double]): Order[PositiveReal] = numOrd.contramap(identity)
    end PositiveReal

    /** Represent the nonnegative subset of real numbers. */
    opaque type NonnegativeReal <: Double = Double

    /** Tools for working with nonnegative real numbers */
    object NonnegativeReal:
        inline def apply(x: Double): NonnegativeReal = 
            inline if x >= 0 then (x: NonnegativeReal)
            else compiletime.error("Negative value where nonnegative is required!")
        def either(x: Double): Either[String, NonnegativeReal] = maybe(x).toRight(s"Cannot refine as nonnegative: $x")
        def maybe(x: Double): Option[NonnegativeReal] = (x >= 0).option{ (x: NonnegativeReal) }
        def unsafe(x: Double): NonnegativeReal = either(x).fold(msg => throw new NumberFormatException(msg), identity)
        given nnRealOrd(using numOrd: Order[Double]): Order[NonnegativeReal] = numOrd.contramap(identity)
    end NonnegativeReal

    enum Delimiter(val sep: String, val ext: String):
        case CommaSeparator extends Delimiter(",", "csv")
        case TabSeparator extends Delimiter("\t", "tsv")

        def canonicalExtension: String = ext
        def join(fields: Array[String]): String = fields mkString sep
        def split(s: String): Array[String] = split(s, -1)
        def split(s: String, limit: Int): Array[String] = s.split(sep, limit)
    end Delimiter

    object Delimiter:
        def fromPath(p: os.Path): Option[Delimiter] = fromExtension(p.ext)
        def fromPathUnsafe = (p: os.Path) => 
            fromPath(p).getOrElse{ throw new IllegalArgumentException(s"Cannot infer delimiter from file: $p") }
        def fromExtension(ext: String): Option[Delimiter] = Delimiter.values.filter(_.ext === ext).headOption
    end Delimiter
    
    final case class FrameIndex(get: NonnegativeInt) extends AnyVal
    object FrameIndex:
        given eqForFrameIndex: Eq[FrameIndex] = Eq.fromUniversalEquals[FrameIndex]
        given showForFrameIndex: Show[FrameIndex] = Show.show(_.get.show)
        def unsafe = NonnegativeInt.unsafe.andThen(FrameIndex.apply)
    end FrameIndex

    final case class PositionIndex(get: NonnegativeInt) extends AnyVal
    object PositionIndex:
        given eqForPositionIndex: Eq[PositionIndex] = Eq.fromUniversalEquals[PositionIndex]
        given showForPositionIndex: Show[PositionIndex] = Show.show(_.get.show)
        def fromInt = NonnegativeInt.either.fmap(_.map(PositionIndex.apply))
        def unsafe = NonnegativeInt.unsafe.andThen(PositionIndex.apply)
    end PositionIndex

    final case class ProbeName(get: String)
    object ProbeName:
        given showForProbeName: Show[ProbeName] = Show.show(_.get)

    final case class RegionId(get: FrameIndex):
        def toInt: NonnegativeInt = get.get
    end RegionId
    object RegionId:
        def fromInt(z: Int) = NonnegativeInt.either(z).map(fromNonnegative)
        def fromNonnegative = RegionId.apply `compose` FrameIndex.apply
    end RegionId

    final case class RoiIndex(get: NonnegativeInt) extends AnyVal
    object RoiIndex:
        implicit val showForRoiIndex: Show[RoiIndex] = Show.show(_.get.show)
        def unsafe = NonnegativeInt.unsafe.andThen(RoiIndex.apply)
    end RoiIndex

    final case class TraceId(get: NonnegativeInt) extends AnyVal
    object TraceId:
        def fromInt = NonnegativeInt.either.fmap(_.map(TraceId.apply))
    end TraceId
    

    /**
      * Write a mapping, from position and frame pair to value, to JSON.
      *
      * @param vKey The key to use for the {@code V} element in each object
      * @param pfToV The mapping of data to write
      * @param writeV How to write each {@code V} element as JSON
      * @return A JSON array of object corresponding to each element of the map
      */
    def posFrameMapToJson[V](vKey: String, pfToV: Map[(PositionIndex, FrameIndex), V])(using writeV: (V) => ujson.Value): ujson.Value = {
        val proc1 = (pf: (PositionIndex, FrameIndex), v: V) => ujson.Obj(
            "position" -> pf._1.get,
            "frame" -> pf._2.get,
            vKey -> writeV(v)
        )
        pfToV.toList.map(proc1.tupled)
    }
}
