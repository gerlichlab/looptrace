package at.ac.oeaw.imba.gerlich.looptrace.space

import scala.util.NotGiven
import cats.Order

/** A point in space */
sealed trait Coordinate { def get: Double }

/** Helpers for working with points in space */
object Coordinate:
    /** Ordering is only among coordinates along the same axis/dimension, and is by the wrapped value. */
    given orderForCoordinate[C <: Coordinate : [C] =>> NotGiven[C =:= Coordinate]]: Order[C] = Order.by(_.get)
end Coordinate

/** Notion of point along x-axis / x-dimension in a space */
final case class XCoordinate(get: Double) extends Coordinate

/** Helpers for working with coordinates along the x-axis / x-dimension */
object XCoordinate:
    given orderForX: Order[XCoordinate] = Order.by(_.get)
end XCoordinate

/** Notion of point along y-axis / y-dimension in a space */
final case class YCoordinate(get: Double) extends Coordinate

/** Helpers for working with coordinates along the y-axis / y-dimension */
object YCoordinate:
    given orderForY: Order[YCoordinate] = Order.by(_.get)

/** Notion of point along z-axis / z-dimension in a space */
final case class ZCoordinate(get: Double) extends Coordinate

/** Helpers for working with coordinates along the z-axis / z-dimension */
object ZCoordinate:
    given orderForZ: Order[ZCoordinate] = Order.by(_.get)