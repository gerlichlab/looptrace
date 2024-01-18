package at.ac.oeaw.imba.gerlich.looptrace

import cats.syntax.eq.*
import cats.syntax.functor.*
import org.scalacheck.Arbitrary
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should

/** Tests for region ID wrapper type */
class TestRegionId extends AnyFunSuite, LooptraceSuite, RefinementWrapperSuite, ScalacheckSuite, should.Matchers:
    test("RegionId.fromInt works; region ID must be nonnegative.") {
        forAll { (z: Int) => RegionId.fromInt(z) match {
            case Left(msg) if z < 0 => msg shouldEqual s"Cannot refine as nonnegative: $z"
            case Right(rid) if z >= 0 => 
                rid.toInt shouldEqual z
                rid.get shouldEqual Timepoint.unsafe(z)
            case result => fail(s"Unexpected result parsing region ID from int ($z): $result")
        } }
    }

    test("RegionId.fromNonnegative works; region ID must be nonnegative.") {
        forAll { (z: NonnegativeInt) => 
            val rid = RegionId.fromNonnegative(z)
            rid.toInt shouldEqual z
            rid.get shouldEqual Timepoint(z)
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
