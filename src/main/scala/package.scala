package at.ac.oeaw.imba.gerlich

import scala.util.Try
import upickle.default.*
import cats.*
import cats.data.*
import cats.derived.*
import cats.syntax.all.*
import mouse.boolean.*
import scopt.Read

import upickle.default.{ Reader as JsonReader }
import io.github.iltotore.iron.:|
import io.github.iltotore.iron.constraint.char.*

import at.ac.oeaw.imba.gerlich.gerlib.imaging.{ ImagingTimepoint, PositionName }
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.nonnegativeInt.given // Order, Show
import at.ac.oewa.imba.gerlich.looptrace.RowIndexAdmission
import at.ac.oeaw.imba.gerlich.looptrace.syntax.json.*

/** Chromatin fiber tracing with FISH probes */
package object looptrace {
    type ErrorMessages = NonEmptyList[String]
    type ErrMsgsOr[A] = Either[ErrorMessages, A]

    def tryToInt(x: Double): Either[String, Int] = {
        val z = x.toInt
        (x == z).either(s"Cannot convert to integer: $x", z) // == rather than === here to allow Double/Int comparison
    }

    extension [A](arr: Array[A])
        private def lookup(a: A): Option[Int] = arr.indexOf(a) match {
            case -1 => None
            case i => i.some
        }
    
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
    
    final case class LocusId(get: ImagingTimepoint) derives Order
    
    object LocusId:
        def fromInt = NonnegativeInt.either.fmap(_.map(fromNonnegative))
        def fromNonnegative = LocusId.apply `compose` ImagingTimepoint.apply
        def unsafe = fromNonnegative `compose` NonnegativeInt.unsafe
    end LocusId

    final case class ProbeName(get: String)

    final case class RegionId(get: ImagingTimepoint) derives Order

    object RegionId:
        def fromInt = NonnegativeInt.either.fmap(_.map(fromNonnegative))
        
        def fromNonnegative = RegionId.apply `compose` ImagingTimepoint.apply
        
        def unsafe = fromNonnegative `compose` NonnegativeInt.unsafe

        given JsonReader[RegionId] = 
            upickle.default.reader[ujson.Value].map(
                _.safeInt.leftMap(_.getMessage).flatMap(fromInt) match {
                    case Left(msg) => throw ujson.IncompleteParseException(msg)
                    case Right(regId) => regId
                }
            )
    end RegionId

    final case class RoiIndex(get: NonnegativeInt) derives Order
    
    object RoiIndex:
        def fromInt = NonnegativeInt.either.fmap(_.map(RoiIndex.apply))
        
        def unsafe = NonnegativeInt.unsafe.andThen(RoiIndex.apply)
        
        given JsonReader[RoiIndex] = 
            upickle.default.reader[ujson.Value].map(
                _.safeInt.leftMap(_.getMessage).flatMap(fromInt) match {
                    case Left(msg) => throw ujson.IncompleteParseException(msg)
                    case Right(roiIdx) => roiIdx
                }
            )
        
        given RowIndexAdmission[RoiIndex, Id] with
            override def getRowIndex: RoiIndex => Id[NonnegativeInt] = _.get
    end RoiIndex

    final case class TraceId(get: NonnegativeInt) derives Order
    
    object TraceId:
        def fromInt = NonnegativeInt.either.fmap(_.map(TraceId.apply))
        def fromRoiIndex(i: RoiIndex): TraceId = new TraceId(i.get)
        def unsafe = NonnegativeInt.unsafe.andThen(TraceId.apply)
    end TraceId
}
