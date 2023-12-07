package at.ac.oeaw.imba.gerlich.looptrace

import scala.math.{ abs, min, pow, sqrt }
import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import at.ac.oeaw.imba.gerlich.looptrace.space.*

/** Tests for distance thresholds */
class TestDistanceThresholds extends AnyFunSuite, LooptraceSuite, ScalacheckSuite, should.Matchers:

    type PointProximity = ProximityComparable[Point3D]

    test("Positive: Disjunctive component proximity implies Euclidean proximity.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        def genPiecewiseProximalPoints: Gen[(NonnegativeReal, Point3D, Point3D)] = {
            for {
                (a, b) <- {
                    given arbRawCoord: Arbitrary[Double] = genReasonableCoordinate.toArbitrary // Prevent Euclidean overflow.
                    arbitrary[(Point3D, Point3D)]
                }
                tMin = minDistPiecewise(a, b) + 1
                t <- Gen.choose(tMin, Double.MaxValue).map(NonnegativeReal.unsafe)
            } yield (t, a, b)
        }

        forAll (genPiecewiseProximalPoints) { case (t, a, b) =>
            val (piecewise, euclidean) = t.toThresholdPair.mapBoth(_.toProximityComparable)
            piecewise.proximal(a, b) shouldBe true
            euclidean.proximal(a, b) shouldBe true
        }
    }

    test("Threshold of 0 always gives false for proximity.") {
        given arbitraryCoordinate: Arbitrary[Double] = genReasonableCoordinate.toArbitrary // Prevent Euclidean overflow.
        forAll { (a: Point3D, b: Point3D) => 
            val (piecewise, euclidean) = NonnegativeReal(0.0).toThresholdPair.mapBoth(_.toProximityComparable)
            /* First, test each strategy with the 2 different points. */
            piecewise.proximal(a, b) shouldBe false
            euclidean.proximal(a, b) shouldBe false
            /* Then, test each strategy with the identical points. */
            piecewise.proximal(a, a) shouldBe false
            euclidean.proximal(a, a) shouldBe false
            piecewise.proximal(b, b) shouldBe false
            euclidean.proximal(b, b) shouldBe false
        }
    }

    test("A point is always proximal with itself when the threshold is positive.") {
        forAll (Gen.posNum[Double].map(NonnegativeReal.unsafe), arbitrary[Point3D]) { case (t, p) => 
            val (piecewise, euclidean) = t.toThresholdPair.mapBoth(_.toProximityComparable)
            val pCopy = p.copy()
            piecewise.proximal(p, p) shouldBe true
            piecewise.proximal(p, pCopy) shouldBe true
            euclidean.proximal(p, p) shouldBe true
            euclidean.proximal(p, pCopy) shouldBe true
        }
    }

    extension [A](g: Gen[A])
        def toArbitrary: Arbitrary[A] = Arbitrary(g)

    extension (t: DistanceThreshold)
        def toProximityComparable: ProximityComparable[Point3D] = DistanceThreshold.defineProximityPointwise(t)

    extension (t: NonnegativeReal)
        def toThresholdPair = PiecewiseDistance.DisjunctiveThreshold(t) -> EuclideanDistance.Threshold(t)
    
    // Prevent overflow in Euclidean.
    private def genReasonableCoordinate: Gen[Double] = Gen.choose(-1e16, 1e16)

    private def minDistPiecewise(a: Point3D, b: Point3D): Double = 
        List(a.x.get - b.x.get, a.y.get - b.y.get, a.z.get - b.z.get).map(abs).sorted.head
end TestDistanceThresholds
