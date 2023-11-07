package at.ac.oeaw.imba.gerlich.looptrace.space

import scala.util.Try
import cats.data.{ NonEmptyList as NEL }
import cats.syntax.either.*
import upickle.default.*

/** Typesafe representation of a point in 3D space */
final case class Point3D(x: XCoordinate, y: YCoordinate, z: ZCoordinate)

/** Designate the ordering of coordinates for a point (e.g. xyz for Forward or zyx for Reverse). */
enum CoordinateSequence derives ReadWriter:
    case Forward, Reverse

object CoordinateSequence:
    /** Conversion of a coordinate sequence member value to JSON value, routing through text representation */
    given toJson(using liftStr: String => ujson.Value): (CoordinateSequence => ujson.Value) = cs => liftStr(cs.toString)

    /** Try to parse given text as enum instance. */
    def parse(s: String): Either[Throwable, CoordinateSequence] = Try{ CoordinateSequence.valueOf(s) }.toEither

    /** Try to parse given JSON value as enum instance. */
    def fromJsonSafe(json: ujson.Value): Either[Throwable, CoordinateSequence] = Try(json.str).toEither.flatMap(parse)

/** Helpers for working with points in 3D space */
object Point3D:
    import CoordinateSequence.*

    /** Try to parse a list of coordinates into a point, failing if wrong dimensionality. */
    def fromList(coordseq: CoordinateSequence)(pt: List[Double]): Either[String, Point3D] = {
        pt match {
            case _1 :: _2 :: _3 :: Nil => {
                val y = YCoordinate(_2)
                val (x, z) = coordseq match {
                    case Forward => XCoordinate(_1) -> ZCoordinate(_3)
                    case Reverse => XCoordinate(_3) -> ZCoordinate(_1)
                }
                Point3D(x, y, z).asRight
            }
            case _ => s"Expected 3 coordinates for a 3D point, but got ${pt.length}".asLeft
        }
    }

    /** Represent the point's coordinates as a list, based on the ordering/sequence given. */
    def toList(coordseq: CoordinateSequence)(pt: Point3D): NEL[Coordinate] = {
        val (head, tail) = coordseq match {
            case Forward => (pt.x, pt.z)
            case Reverse => (pt.z, pt.x)
        }
        NEL(head, List(pt.y, tail))
    }
