package at.ac.oeaw.imba.gerlich.looptrace
package space

import scala.util.{ NotGiven, Try }
import cats.*
import cats.syntax.all.*

import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

import at.ac.oeaw.imba.gerlich.looptrace.space.{ Coordinate, XCoordinate, YCoordinate, ZCoordinate }

/** Bundle the 3 intervals that define a rectangular prism in 3D. */
final case class BoundingBox(
    sideX: BoundingBox.Interval[XCoordinate], 
    sideY: BoundingBox.Interval[YCoordinate], 
    sideZ: BoundingBox.Interval[ZCoordinate]
    )

/** Helpers for working with the notion of a 3D bounding box */
object BoundingBox:
    given orderForBoundingBox: Order[BoundingBox] = Order.by{ 
        case BoundingBox(
            BoundingBox.Interval(XCoordinate(loX), XCoordinate(hiX)), 
            BoundingBox.Interval(YCoordinate(loY), YCoordinate(hiY)), 
            BoundingBox.Interval(ZCoordinate(loZ), ZCoordinate(hiZ))
        ) => (loX, loY, loZ, hiX, hiY, hiZ)
    }

    /** A margin for an expansion (e.g. an interval) around a point */
    private[looptrace] final case class Margin(get: NonnegativeReal) extends AnyVal

    /**
      * An 1D interval is defined by its endpoints.
      *
      * @tparam C The type of [[at.ac.oeaw.imba.gerlich.looptrace.space.Coordinate]] stored in 
      *     each endpoint, essentially the type of field on which the interval is defined
      * @param lo The lower bound of the interval
      * @param hi The upper bound of the interval
      */
    final case class Interval[C <: Coordinate : [C] =>> NotGiven[C =:= Coordinate]](lo: C, hi: C):
        import Coordinate.given
        require(lo < hi, s"Lower bound not less than upper bound: ($lo, $hi)")
    end Interval

    object Interval:
        def fromTuple[C <: Coordinate : [C] =>> NotGiven[C =:= Coordinate]](t: (C, C)): Either[String, Interval[C]] = 
            Try{ new Interval(t._1, t._2) }
                .toEither
                .leftMap{ e => s"Failed to create interval: ${e.getMessage}" }
end BoundingBox
