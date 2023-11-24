package at.ac.oeaw.imba.gerlich.looptrace.space

import scala.util.NotGiven
import cats.Order

sealed trait Coordinate { def get: Double }
object Coordinate:
    given orderForCoordinate[C <: Coordinate : [C] =>> NotGiven[C =:= Coordinate]]: Order[C] = Order.by(_.get)

final case class XCoordinate(get: Double) extends Coordinate
object XCoordinate:
    given orderForX: Order[XCoordinate] = Order.by(_.get)

final case class YCoordinate(get: Double) extends Coordinate
object YCoordinate:
    given orderForY: Order[YCoordinate] = Order.by(_.get)
final case class ZCoordinate(get: Double) extends Coordinate
object ZCoordinate:
    given orderForZ: Order[ZCoordinate] = Order.by(_.get)