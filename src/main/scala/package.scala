package at.ac.oeaw.imba.gerlich

import scala.util.Try
import upickle.default.*
import cats.*
import cats.data.*
import cats.derived.*
import cats.syntax.all.*
import mouse.boolean.*
import scopt.Read
import com.github.tototoshi.csv.*

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt.given // Order, Show

/** Chromatin fiber tracing with FISH probes */
package object looptrace {
    val VersionName = "0.6.0"

    type CsvRow = Map[String, String]
    type ErrorMessages = NonEmptyList[String]
    type ErrMsgsOr[A] = Either[ErrorMessages, A]

    /** Use rows from a CSV file in arbitrary code. */
    def withCsvData(filepath: os.Path)(code: Iterable[CsvRow] => Any): Any = {
        val reader = CSVReader.open(filepath.toIO)
        try { code(reader.allWithHeaders()) } finally { reader.close() }
    }

    /** Do arbitrary code with rows from a pair of CSV files. */
    def withCsvPair(f1: os.Path, f2: os.Path)(code: (Iterable[CsvRow], Iterable[CsvRow]) => Any): Any = {
        var reader1: CSVReader = null
        val reader2 = CSVReader.open(f2.toIO)
        try {
            reader1 = CSVReader.open(f1.toIO)
            code(reader1.allWithHeaders(), reader2.allWithHeaders())
        } finally {
            if (reader1 != null) { reader1.close() }
            reader2.close()
        }
    }

    extension [A, F[_, _] : Bifunctor](faa: F[A, A])
        def mapBoth[B](f: A => B): F[B, B] = faa.bimap(f, f)

    /** When an iterable is all booleans, simplify the all-true check ({@code ps.forall(identity) === ps.all}) */
    extension (ps: Iterable[Boolean])
        def all: Boolean = ps.forall(identity)
        def any: Boolean = ps.exists(identity)

    /** Add a {@code .parent} accessor on a path. */
    extension (p: os.Path)
        def parent: os.Path = p / os.up

    def tryToInt(x: Double): Either[String, Int] = {
        val z = x.toInt
        (x == z).either(s"Cannot convert to integer: $x", z) // == rather than === here to allow Double/Int comparison
    }
    
    extension (v: ujson.Value)
        def int: Int = tryToInt(v.num).fold(msg => throw new ujson.Value.InvalidData(v, msg), identity)

    extension (v: ujson.Value)
        def safeInt = Try{ v.int }.toEither

    extension [A](arr: Array[A])
        def lookup(a: A): Option[Int] = arr.indexOf(a) match {
            case -1 => None
            case i => i.some
        }

    extension [I, O](f: I => Unit)
        def returning(o: O): I => O = f `andThen` Function.const(o)

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

    object PositiveIntExtras:
        def lengthOfNonempty(xs: NonEmptyList[?]): PositiveInt = PositiveInt.unsafe(xs.length)
        def lengthOfNonempty[A : Order](xs: NonEmptySet[A]): PositiveInt = PositiveInt.unsafe(xs.length)
    end PositiveIntExtras
    
    /** Type wrapper around the index of an imaging channel */
    final case class Channel(get: NonnegativeInt) derives Order
    
    /** Helpers for working with the representation of an imaging channel */
    object Channel:
        def fromInt = NonnegativeInt.either.fmap(_.map(Channel.apply))
        def unsafe = NonnegativeInt.unsafe `andThen` Channel.apply
    end Channel

    final case class LocusId(get: ImagingTimepoint) derives Order:
        def index = get.get
    
    object LocusId:
        def fromInt = NonnegativeInt.either.fmap(_.map(fromNonnegative))
        def fromNonnegative = LocusId.apply `compose` ImagingTimepoint.apply
        def unsafe = fromNonnegative `compose` NonnegativeInt.unsafe
    end LocusId

    final case class PositionIndex(get: NonnegativeInt) derives Order
    
    object PositionIndex:
        def fromInt = NonnegativeInt.either.fmap(_.map(PositionIndex.apply))
        def unsafe = NonnegativeInt.unsafe `andThen` PositionIndex.apply
    end PositionIndex

    final case class PositionName(get: String) derives Order

    final case class ProbeName(get: String)

    final case class RegionId(get: ImagingTimepoint) derives Order:
        def index = get.get

    object RegionId:
        def fromInt = NonnegativeInt.either.fmap(_.map(fromNonnegative))
        def fromNonnegative = RegionId.apply `compose` ImagingTimepoint.apply
        def unsafe = fromNonnegative `compose` NonnegativeInt.unsafe
    end RegionId

    final case class RoiIndex(get: NonnegativeInt) derives Order
    
    object RoiIndex:
        def fromInt = NonnegativeInt.either.fmap(_.map(RoiIndex.apply))
        def unsafe = NonnegativeInt.unsafe.andThen(RoiIndex.apply)
    end RoiIndex

    final case class TraceId(get: NonnegativeInt) derives Order
    
    object TraceId:
        def fromInt = NonnegativeInt.either.fmap(_.map(TraceId.apply))
        def fromRoiIndex(i: RoiIndex): TraceId = new TraceId(i.get)
        def unsafe = NonnegativeInt.unsafe.andThen(TraceId.apply)
    end TraceId

    /**
      * Write a mapping, from position and time pair to value, to JSON.
      *
      * @param vKey The key to use for the {@code V} element in each object
      * @param ptToV The mapping of data to write
      * @param writeV How to write each {@code V} element as JSON
      * @return A JSON array of object corresponding to each element of the map
      */
    def posTimeMapToJson[V](vKey: String, ptToV: Map[(PositionIndex, ImagingTimepoint), V])(using writeV: (V) => ujson.Value): ujson.Value = {
        val proc1 = (pt: (PositionIndex, ImagingTimepoint), v: V) => ujson.Obj(
            "position" -> pt._1.get,
            "timepoint" -> pt._2.get,
            vKey -> writeV(v)
        )
        ptToV.toList.map(proc1.tupled)
    }
}
