package at.ac.oeaw.imba.gerlich.looptrace.space

import scala.math.{ pow, sqrt }
import cats.*
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.{ NonnegativeInt, NonnegativeReal }
import at.ac.oeaw.imba.gerlich.looptrace.{ all, any }

/** Something that can compare two {@code A} values w.r.t. threshold value of type {@code T} */
trait ProximityComparable[A]:
    /** Are the two {@code A} values within threshold {@code T} of each other? */
    def proximal: (A, A) => Boolean
end ProximityComparable

/** Helpers for working with proximity comparisons */
object ProximityComparable:
    extension [A](a1: A)(using ev: ProximityComparable[A])
        infix def proximal(a2: A): Boolean = ev.proximal(a1, a2)

    given contravariantForProximityComparable: Contravariant[ProximityComparable] = 
        new Contravariant[ProximityComparable] {
            override def contramap[A, B](fa: ProximityComparable[A])(f: B => A) = new ProximityComparable[B] {
                override def proximal = (b1, b2) => fa.proximal(f(b1), f(b2))
            }
        }
end ProximityComparable

/** A threshold on distances, which should be nonnegative, to be semantically contextualised by the subtype */
sealed trait DistanceThreshold{ def get: NonnegativeReal }

/** Helpers for working with distance thresholds */
object DistanceThreshold:
    given showForDistanceThreshold: Show[DistanceThreshold] = Show.show{ (t: DistanceThreshold) => 
        val typeName = t match {
            case _: EuclideanDistance.Threshold => "Euclidean"
            case _: PiecewiseDistance.ConjunctiveThreshold => "Conjunctive"
        }
        s"${typeName}Threshold(${t.get})"
    }

    def defineProximityPointwise(threshold: DistanceThreshold): ProximityComparable[Point3D] = threshold match {
        case t: EuclideanDistance.Threshold => new ProximityComparable[Point3D] {
            override def proximal = (a, b) => 
                val d = EuclideanDistance.between(a, b)
                if (d.isInfinite) { throw new EuclideanDistance.OverflowException(s"Cannot compute finite distance between $a and $b") }
                d `lt` t
        }
        case t: PiecewiseDistance.ConjunctiveThreshold => 
            new ProximityComparable[Point3D] {
                override def proximal = PiecewiseDistance.within(t)
            }
    }

    def defineProximityPointwise[A](threshold: DistanceThreshold): (A => Point3D) => ProximityComparable[A] = 
        defineProximityPointwise(threshold).contramap
end DistanceThreshold

/** Helpers for working with distances in by-component / piecewise fashion */
object PiecewiseDistance:
    
    /** Distance threshold in which predicate comparing values to this threshold operates conjunctively over components */
    final case class ConjunctiveThreshold(get: NonnegativeReal) extends DistanceThreshold

    /** Are points closer than given threshold along any axis? */
    def within(threshold: ConjunctiveThreshold)(a: Point3D, b: Point3D): Boolean = ((a, b) match {
        case (
            Point3D(XCoordinate(x1), YCoordinate(y1), ZCoordinate(z1)), 
            Point3D(XCoordinate(x2), YCoordinate(y2), ZCoordinate(z2))
        ) => List(x1 - x2, y1 - y2, z1 - z2)
    }).forall(diff => diff.abs < threshold.get)
end PiecewiseDistance

/** Semantic wrapper to denote that a nonnegative real number represents a Euclidean distance */
final case class EuclideanDistance private(get: NonnegativeReal) extends AnyVal:
    final def lessThan(t: EuclideanDistance.Threshold): Boolean = get < t.get
    final def lt = lessThan
    final def greaterThan = !lessThan(_: EuclideanDistance.Threshold)
    final def gt = greaterThan
    final def equalTo(t: EuclideanDistance.Threshold) = !lt(t) && !gt(t)
    final def eq = equalTo
    final def lteq(t: EuclideanDistance.Threshold) = lt(t) || eq(t)
    final def gteq(t: EuclideanDistance.Threshold) = gt(t) || eq(t)
    final def isFinite = get.isFinite
    final def isInfinite = !isFinite
end EuclideanDistance

/** Helpers for working with Euclidean distances */
object EuclideanDistance:
    given orderForEuclDist: Order[EuclideanDistance] = Order.by(_.get)

    /** When something goes wrong with a distance computation or comparison */
    final case class OverflowException(message: String) extends Exception(message)

    /** Comparison basis for Euclidean distance between points */
    final case class Threshold(get: NonnegativeReal) extends DistanceThreshold
    
    // TODO: account for infinity/null-numeric cases.
    def between(a: Point3D, b: Point3D): EuclideanDistance = (a, b) match {
        case (Point3D(x1, y1, z1), Point3D(x2, y2, z2)) => 
            val d = NonnegativeReal.unsafe(
                math.sqrt{ pow(x1.get - x2.get, 2) + pow(y1.get - y2.get, 2) + pow(z1.get - z2.get, 2) }
            )
            new EuclideanDistance(d)
    }
    
    /** Use a lens of a 3D point from arbitrary type {@code A} to compute distance between {@code A} values. */
    def between[A](p: A => Point3D)(a1: A, a2: A): EuclideanDistance = between(p(a1), p(a2))
end EuclideanDistance
