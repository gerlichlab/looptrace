package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import cats.Order
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
