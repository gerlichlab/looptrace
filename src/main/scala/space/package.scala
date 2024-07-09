package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ NotGiven, Try }
import cats.Order
import cats.data.NonEmptyList
import cats.syntax.either.*
import upickle.default.*

import at.ac.oeaw.imba.gerlich.gerlib.geometry

/** Working with (3D) space */
package object space:

    private type Wrapped = Double

    type Coordinate = geometry.Coordinate[Wrapped]

    object Coordinate:
        /** Ordering is only among coordinates along the same axis/dimension, and is by the wrapped value. */
        given orderForCoordinate[C <: Coordinate : [C] =>> NotGiven[C =:= Coordinate]]: Order[C] = Order.by(_.get)

    type XCoordinate = geometry.XCoordinate[Wrapped]

    object XCoordinate:
        def apply(x: Double): XCoordinate = geometry.XCoordinate(x)

    type YCoordinate = geometry.YCoordinate[Wrapped]

    object YCoordinate:
        def apply(y: Double): YCoordinate = geometry.YCoordinate(y)

    type ZCoordinate = geometry.ZCoordinate[Wrapped]

    object ZCoordinate:
        def apply(z: Double): ZCoordinate = geometry.ZCoordinate(z)

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

    type Point3D = geometry.Point3D[Wrapped]

    /** Helpers for working with points in 3D space */
    object Point3D:
        import CoordinateSequence.*

        def apply(x: XCoordinate, y: YCoordinate, z: ZCoordinate): Point3D = 
            geometry.Point3D(x = x, y = y, z = z)

        /** Order component-wise, (z, y, x) */
        given orderForPoint3D: Order[Point3D] = Order.by(pt => (pt.z, pt.y, pt.x))

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
        def toList(coordseq: CoordinateSequence)(pt: Point3D): NonEmptyList[Coordinate] = {
            val (head, tail) = coordseq match {
                case Forward => (pt.x, pt.z)
                case Reverse => (pt.z, pt.x)
            }
            NonEmptyList(head, List(pt.y, tail))
        }