package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import cats.*
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.space.{ Coordinate, XCoordinate, YCoordinate, ZCoordinate }

final case class BoundingBox(
    sideX: BoundingBox.Interval[XCoordinate], 
    sideY: BoundingBox.Interval[YCoordinate], 
    sideZ: BoundingBox.Interval[ZCoordinate]
    )

object BoundingBox:
    given orderForBoundingBox: Order[BoundingBox] = Order.by{ 
        case BoundingBox(
            BoundingBox.Interval(XCoordinate(loX), XCoordinate(hiX)), 
            BoundingBox.Interval(YCoordinate(loY), YCoordinate(hiY)), 
            BoundingBox.Interval(ZCoordinate(loZ), ZCoordinate(hiZ))
        ) => (loX, loY, loZ, hiX, hiY, hiZ)
    }

    /** A margin for an expansion (e.g. an interval) around a point */
    final case class Margin(get: NonnegativeReal) extends AnyVal

    final case class Interval[C <: Coordinate : [C] =>> NotGiven[C =:= Coordinate]](lo: C, hi: C):
        require(lo < hi, s"Lower bound not less than upper bound: ($lo, $hi)")
    end Interval
end BoundingBox
