package at.ac.oeaw.imba.gerlich.looptrace.space

import scala.math.{ pow, sqrt }
import cats.Order
import at.ac.oeaw.imba.gerlich.looptrace.{ NonnegativeInt, NonnegativeReal }

/** Computing distances of type {@code S} between {@code A}s, and comparing to a value of type {@code T} */
trait Metric[A, D, T]:
    def distanceBetween(a1: A, a2: A): D
    protected def within(t: T)(d: D): Boolean
    final def within(t: T)(a1: A, a2: A): Boolean = within(t)(distanceBetween(a1, a2))
end Metric

/** Semantic wrapper to denote that a nonnegative real number represents a Euclidean distance */
final case class EuclideanDistance private(get: NonnegativeReal):
    final def lessThan(t: EuclideanDistance.Threshold): Boolean = get < t.get
    final def lt = lessThan
    final def greaterThan = !lessThan(_: EuclideanDistance.Threshold)
    final def gt = greaterThan
    final def equalTo(t: EuclideanDistance.Threshold) = !lt(t) && !gt(t)
    final def eq = equalTo
    final def lteq(t: EuclideanDistance.Threshold) = lt(t) || eq(t)
    final def gteq(t: EuclideanDistance.Threshold) = gt(t) || eq(t)
end EuclideanDistance

/** Helpers for working with Euclidean distances */
object EuclideanDistance:
    given orderForEuclDist: Order[EuclideanDistance] = Order.by(_.get)

    def getMetric[A](p: A => Point3D): Metric[A, EuclideanDistance, Threshold] = 
        new Metric[A, EuclideanDistance, Threshold] {
            override def distanceBetween(a1: A, a2: A) = byPoints(p)(a1, a2)
            override def within(t: Threshold)(d: EuclideanDistance) = d `lt` t
        }

    final case class Threshold(get: NonnegativeReal) extends AnyVal

    // TODO: account for overflow cases.
    def between(a: Point3D, b: Point3D): EuclideanDistance = (a, b) match {
        case (Point3D(x1, y1, z1), Point3D(x2, y2, z2)) => 
            val d = NonnegativeReal.unsafe(
                math.sqrt{ pow(x1.get - x2.get, 2) + pow(y1.get - y2.get, 2) + pow(z1.get - z2.get, 2) }
            )
            new EuclideanDistance(d)
    }

    /** Use a lens of a 3D point from arbitrary type {@code A} to compute distance between {@code A} values. */
    def byPoints[A](p: A => Point3D) = (a1: A, a2: A) => between(p(a1), p(a2))
end EuclideanDistance
