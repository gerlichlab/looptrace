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
import io.github.iltotore.iron.{ :|, refineEither }
import io.github.iltotore.iron.constraint.any.{ StrictEqual, Not }
import io.github.iltotore.iron.constraint.string.Match
import io.github.iltotore.iron.constraint.collection.{ Empty, ForAll }
import io.github.iltotore.iron.constraint.char.*

import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.imaging.{ ImagingTimepoint, PositionName }
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.json.JsonValueWriter
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.nonnegativeInt.given // Order, Show
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*
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

    private type TraceGroupNameConstraint = Not[Empty] & Not[ForAll[Whitespace]]
    
    private type ValidTraceGroupName = String :| TraceGroupNameConstraint

    object ValidTraceGroupName:
        def either: String => Either[String, ValidTraceGroupName] = _.refineEither[TraceGroupNameConstraint]

    /** A name for a certain kind tracing structure in a particular experiment */
    final case class TraceGroupId(get: ValidTraceGroupName)

    /** Helpers for working with identifiers of tracing groups/structures */
    object TraceGroupId:
        given orderForTraceGroupId(using Order[String]): Order[TraceGroupId] = 
            import io.github.iltotore.iron.cats.given
            Order.by(_.get)

        given JsonValueWriter[TraceGroupId, ujson.Str] with
            override def apply(i: TraceGroupId): ujson.Str = ujson.Str(i.get)
        
        given SimpleShow[TraceGroupId] = SimpleShow.instance(_.get)

        /** The given name is valid if and only if it's nonempty. */
        def fromString: String => Either[String, TraceGroupId] = 
            ValidTraceGroupName.either.fmap(_.map(TraceGroupId.apply))

        /** Use the {@code .fromString} implementation, getting the result or throwing an error. */
        def unsafe: String => TraceGroupId = s => 
            fromString(s)
                .leftMap{ msg => new Exception(s"Illegal value ($s) as trace group ID: $msg") }
                .fold(throw _, identity)
    end TraceGroupId

    /** A trace group ID which may or may not be present, isomorphic to {@code Option[TraceGroupId]} */
    opaque type TraceGroupMaybe = Option[TraceGroupId]

    /** Helpers for working with optional trace group IDs */
    object TraceGroupMaybe:
        def apply(tg: Option[TraceGroupId]): TraceGroupMaybe = tg
        
        def apply(tg: TraceGroupId): TraceGroupMaybe = tg.some
        
        def empty: TraceGroupMaybe = Option.empty
        
        def fromString: String => Either[String, TraceGroupMaybe] = s => 
            if s.isEmpty then empty.asRight
            else TraceGroupId.fromString(s).map(apply)

        given orderForTraceGroupMaybe(using Order[TraceGroupId]): Order[TraceGroupMaybe] = Order.by(_.toOption)

        /** The JSON representation is {@code ujson.Null} exactly when the optional ID is empty. */
        given JsonValueWriter[TraceGroupMaybe, ujson.Str | ujson.Null.type] with
            override def apply(groupOpt: TraceGroupMaybe): ujson.Str | ujson.Null.type = 
                import TraceGroupId.given
                import at.ac.oeaw.imba.gerlich.gerlib.json.syntax.asJson
                groupOpt.toOption.fold(ujson.Null)(_.asJson)

        given SimpleShow[TraceGroupMaybe] = SimpleShow.instance(_.fold("")(_.get))

        extension (tg: TraceGroupMaybe)
            def toOption: Option[TraceGroupId] = tg
    end TraceGroupMaybe

    /** Identifier of a particular trace (unique at the level */
    final case class TraceId(get: NonnegativeInt) derives Order
    
    /** Helpers for working with an identifier of a particular trace */
    object TraceId:
        def fromInt = NonnegativeInt.either.fmap(_.map(TraceId.apply))
        def fromRoiIndex(i: RoiIndex): TraceId = new TraceId(i.get)
        def unsafe = NonnegativeInt.unsafe.andThen(TraceId.apply)
    end TraceId

    // A 1-based count encoding of the field of view, prefixed with "P" for "position", and 4 digits to hold up to 9999 FOVs
    private type OneBasedFourDigitPositionNameConstraint = Match["P\\d{4}"] & Not[StrictEqual["P0000"]]
    
    // A String which complies with the domain restriction for representing a field of view by a 1-based count
    type OneBasedFourDigitPositionName = String :| OneBasedFourDigitPositionNameConstraint
    
    /** Helpers for working with the String refinement representing field of view name */
    object OneBasedFourDigitPositionName:
        /** Attempt to further refine the position name as one in compliance with using 4 digits for a one-based count */
        def fromPositionName: PositionName => Either[String, OneBasedFourDigitPositionName] = 
            ((_: PositionName).show_) `andThen` fromString(true)
        
        private def fromString(trimZarr: Boolean): String => Either[String, OneBasedFourDigitPositionName] = 
            s => (if trimZarr then s.stripSuffix(".zarr") else s).refineEither[OneBasedFourDigitPositionNameConstraint]
    end OneBasedFourDigitPositionName
}
