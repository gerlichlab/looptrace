package at.ac.oeaw.imba.gerlich.looptrace

import scala.math.{ abs, min, pow, sqrt }
import cats.syntax.all.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.geometry.{ DistanceThreshold, EuclideanDistance, PiecewiseDistance, ProximityComparable }
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.looptrace.space.*

/** Tests for distance thresholds */
class TestDistanceThresholds extends AnyFunSuite, ScalaCheckPropertyChecks, DistanceSuite, LooptraceSuite, should.Matchers:
    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)

    /** Compare points in 3D space for proximity. */
    type PointProximity = ProximityComparable[Point3D]
    type PtPair = (Point3D, Point3D)

    test("Positive: Conjunctive component proximity implies disjunctive component proximity.") {
        given [A] => Shrink[A] = Shrink.shrinkAny[A]
        def genProximalPoints: Gen[(NonnegativeReal, Point3D, Point3D)] = {
            for
                (a, b) <- {
                    given arbRawCoord: Arbitrary[Double] = genReasonableCoordinate.toArbitrary // Prevent Euclidean overflow.
                    arbitrary[(Point3D, Point3D)]
                }
                // Generate a threshold value that's 1 greater than the greatest component difference.
                tMin = {
                    val dist = PiecewiseDistance.between(a, b)
                    1 + List(dist.getX, dist.getY, dist.getZ).max
                }
                t <- Gen.choose(tMin, Double.MaxValue).map(NonnegativeReal.unsafe)
            yield (t, a, b)
        }

        forAll (genProximalPoints) { case (t, a, b) =>
            val thresholds = List(PiecewiseDistance.ConjunctiveThreshold(t))
            forAll (Table("comparison", thresholds.map(_.toProximityComparable)*)) { _.proximal(a, b) shouldBe true }
        }
    }

    test("Threshold of 0 always gives false for proximity.") {
        given arbitraryCoordinate: Arbitrary[Double] = genReasonableCoordinate.toArbitrary // Prevent Euclidean overflow.
        val comparisonsStrategies = NonnegativeReal(0.0).toAllProximityComparables
        val pointUseStrategies: List[PtPair => PtPair] = List(p => p._1 -> p._1, p => p._1 -> p._2, p => p._2 -> p._2)
        forAll (Table(
            "comparisonAndPointUse", 
            comparisonsStrategies.flatMap(cmp => pointUseStrategies.map(cmp -> _))*)) { 
            case (cmp, getPts) => forAll { (pts: PtPair) => cmp.proximal.tupled(pts) shouldBe false }
        }
    }

    test("A point is always proximal with itself when the threshold is positive.") {
        val pointPairStrategies: List[Point3D => PtPair] = List(p => p -> p, p => p -> p.copy())
        forAll (Gen.posNum[Double].map(NonnegativeReal.unsafe), arbitrary[Point3D]) { case (t, p) => 
            forAll (Table(
                "comparisonAndPointUse", 
                t.toAllProximityComparables.flatMap{ cmp => pointPairStrategies.map(cmp -> _) }*
            )) { case (cmp, toPair) => cmp.proximal.tupled(toPair(p)) shouldBe true }
        }
    }

    test("Each type proximity gets examples right.") {
        pending
    }

    extension (t: DistanceThreshold)
        def toProximityComparable: ProximityComparable[Point3D] = DistanceThreshold.defineProximityPointwise(t)

    extension (t: NonnegativeReal)
        def toAllProximityComparables = List(
            PiecewiseDistance.ConjunctiveThreshold(t).toProximityComparable,
            EuclideanDistance.Threshold(t).toProximityComparable
        )
end TestDistanceThresholds
