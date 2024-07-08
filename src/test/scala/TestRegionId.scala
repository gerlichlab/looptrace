package at.ac.oeaw.imba.gerlich.looptrace

import cats.syntax.all.*
import org.scalacheck.Arbitrary
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Tests for region ID wrapper type */
class TestRegionId extends AnyFunSuite, ScalaCheckPropertyChecks, LooptraceSuite, RefinementWrapperSuite, should.Matchers:
    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)
    
    test("RegionId.fromInt works; region ID must be nonnegative.") {
        forAll { (z: Int) => RegionId.fromInt(z) match {
            case Left(msg) if z < 0 => msg shouldEqual s"Cannot refine as nonnegative: $z"
            case Right(rid@RegionId(ImagingTimepoint(t))) if z >= 0 => 
                t shouldEqual z
                rid.get shouldEqual ImagingTimepoint.unsafe(z)
            case result => fail(s"Unexpected result parsing region ID from int ($z): $result")
        } }
    }

    test("RegionId.fromNonnegative works; region ID must be nonnegative.") {
        forAll { (z: NonnegativeInt) => 
            val rid = RegionId.fromNonnegative(z)
            rid.get shouldEqual ImagingTimepoint(z)
            rid.index shouldEqual z
        }
    }

    test("Region IDs are equivalent on their wrapped values.") {
        forAll (genEquivalenceInputAndExpectation(RegionId.apply)) { case (f1, f2, exp) => f1 === f2 shouldBe exp }
    }

    test("Set respects region ID equivalence.") {
        forAll (genValuesAndNumUnique(arbitrary[NonnegativeInt])(RegionId.fromNonnegative)) { 
            case (ids, exp) => ids.toSet shouldEqual exp
        }
    }
end TestRegionId
