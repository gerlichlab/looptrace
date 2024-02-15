package at.ac.oeaw.imba.gerlich

import scala.util.Try
import upickle.default.*
import cats.*
import cats.data.*
import cats.syntax.all.*
import mouse.boolean.*
import scopt.Read
import com.github.tototoshi.csv.*

/** Chromatin fiber tracing with FISH probes */
package object looptrace {
    val VersionName = "0.2.0-SNAPSHOT"

    type CsvRow = Map[String, String]
    type ErrorMessages = NonEmptyList[String]
    type ErrMsgsOr[A] = Either[ErrorMessages, A]
    
    /** Nonempty set wrapped in {@code Right} if no duplicates, or {@code Left}-wrapped pairs of element and repeat count */
    def safeNelToNes[A : Order](xs: NonEmptyList[A]): Either[NonEmptyList[(A, Int)], NonEmptySet[A]] = {
        val histogram = xs.groupByNem(identity).toNel.map(_.map(_.size))
        histogram.filter(_._2 > 1).toNel.toLeft(histogram.map(_._1).toNes)
    }

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

    /** Wrapper around {@code os.write} to handle writing an iterable of lines. */
    def writeTextFile(target: os.Path, data: Iterable[Array[String]], delimiter: Delimiter) = 
        os.write(target, data.map(delimiter.join(_: Array[String]) ++ "\n"))

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

    /** Allow custom types as CLI parameters. */
    object ScoptCliReaders:
        given pathRead(using fileRead: Read[java.io.File]): Read[os.Path] = fileRead.map(os.Path.apply)
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
        def add(n1: NonnegativeInt, n2: NonnegativeInt): NonnegativeInt = unsafe(n1 + n2)
        def either(z: Int): Either[String, NonnegativeInt] = maybe(z).toRight(s"Cannot refine as nonnegative: $z")
        def indexed[A](xs: List[A]): List[(A, NonnegativeInt)] = {
            // guaranteed nonnegative by construction here
            xs.zipWithIndex.map{ case (x, i) => x -> unsafe(i) }
        }
        def maybe(z: Int): Option[NonnegativeInt] = (z >= 0).option{ (z: NonnegativeInt) }
        def seqTo(n: NonnegativeInt): IndexedSeq[NonnegativeInt] = (0 to n).map(unsafe)
        def unsafe(z: Int): NonnegativeInt = either(z).fold(msg => throw new NumberFormatException(msg), identity)
        given nonnegativeIntOrder(using intOrd: Order[Int]): Order[NonnegativeInt] = intOrd.contramap(identity)
        given showForNonnegativeInt: Show[NonnegativeInt] = Show.fromToString[NonnegativeInt]
    end NonnegativeInt

    /** Refinement type for nonnegative integers */
    opaque type PositiveInt <: Int = Int
    
    /** Helpers for working with nonnegative integers */
    object PositiveInt:
        inline def apply(z: Int): PositiveInt = 
            inline if z <= 0 then compiletime.error("Non-positive integer where positive is required!")
            else (z: PositiveInt)
        extension (n: PositiveInt)
            def asNonnegative: NonnegativeInt = NonnegativeInt.unsafe(n)
        def either(z: Int): Either[String, PositiveInt] = maybe(z).toRight(s"Cannot refine as positive: $z")
        def maybe(z: Int): Option[PositiveInt] = (z > 0).option{ (z: PositiveInt) }
        def unsafe(z: Int): PositiveInt = either(z).fold(msg => throw new NumberFormatException(msg), identity)
        given posIntOrder(using intOrd: Order[Int]): Order[PositiveInt] = intOrd.contramap(identity)
        given posIntShow(using intShow: Show[Int]): Show[PositiveInt] = intShow.contramap(identity)
        given posIntRW(using intRW: ReadWriter[Int]): ReadWriter[PositiveInt] = intRW.bimap(identity, _.int)
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
    
    /** Type wrapper around the index of an imaging channel */
    final case class Channel(get: NonnegativeInt) extends AnyVal
    
    /** Helpers for working with the representation of an imaging channel */
    object Channel:
        /** Order channels by the wrapped value. */
        given orderForChannel: Order[Channel] = Order.by(_.get)
        def fromInt = NonnegativeInt.either.fmap(_.map(Channel.apply))
        def unsafe = NonnegativeInt.unsafe `andThen` Channel.apply
    end Channel

    final case class LocusId(get: Timepoint):
        def index = get.get
    object LocusId:
        given orderForLocusId: Order[LocusId] = Order.by(_.get)
        given showForLocusId: Show[LocusId] = Show.show(_.index.show)
        def fromInt = NonnegativeInt.either.fmap(_.map(fromNonnegative))
        def fromNonnegative = LocusId.apply `compose` Timepoint.apply
        def unsafe = fromNonnegative `compose` NonnegativeInt.unsafe
    end LocusId

    final case class PositionIndex(get: NonnegativeInt) extends AnyVal
    object PositionIndex:
        given orderForPositionIndex: Order[PositionIndex] = Order.by(_.get)
        given showForPositionIndex: Show[PositionIndex] = Show.show(_.get.show)
        def fromInt = NonnegativeInt.either.fmap(_.map(PositionIndex.apply))
        def unsafe = NonnegativeInt.unsafe `andThen` PositionIndex.apply
    end PositionIndex

    final case class PositionName(get: String) extends AnyVal
    object PositionName:
        given orderForPositionName: Order[PositionName] = Order.by(_.get)
        given showForPositionName: Show[PositionName] = Show.show(_.get)
    end PositionName

    final case class ProbeName(get: String)
    object ProbeName:
        given showForProbeName: Show[ProbeName] = Show.show(_.get)

    final case class RegionId(get: Timepoint):
        def index = get.get
    object RegionId:
        given orderForRegionId: Order[RegionId] = Order.by(_.get)
        given showForRegionId: Show[RegionId] = Show.show(_.index.show)
        def fromInt = NonnegativeInt.either.fmap(_.map(fromNonnegative))
        def fromNonnegative = RegionId.apply `compose` Timepoint.apply
        def unsafe = fromNonnegative `compose` NonnegativeInt.unsafe
    end RegionId

    final case class RoiIndex(get: NonnegativeInt) extends AnyVal
    object RoiIndex:
        given orderForRoiIndex: Order[RoiIndex] = Order.by(_.get)
        given showForRoiIndex: Show[RoiIndex] = Show.show(_.get.show)
        def fromInt = NonnegativeInt.either.fmap(_.map(RoiIndex.apply))
        def unsafe = NonnegativeInt.unsafe.andThen(RoiIndex.apply)
    end RoiIndex

    final case class TraceId(get: NonnegativeInt) extends AnyVal
    object TraceId:
        given orderForTraceId: Order[TraceId] = Order.by(_.get)
        given showForTraceId: Show[TraceId] = Show.show(_.get.toString)
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
    def posTimeMapToJson[V](vKey: String, ptToV: Map[(PositionIndex, Timepoint), V])(using writeV: (V) => ujson.Value): ujson.Value = {
        val proc1 = (pt: (PositionIndex, Timepoint), v: V) => ujson.Obj(
            "position" -> pt._1.get,
            "timepoint" -> pt._2.get,
            vKey -> writeV(v)
        )
        ptToV.toList.map(proc1.tupled)
    }
}
