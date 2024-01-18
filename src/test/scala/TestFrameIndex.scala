package at.ac.oeaw.imba.gerlich.looptrace

import cats.syntax.eq.*
import org.scalacheck.Gen
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should

/** Tests for frame index wrapper type */
class TestTimepoint extends AnyFunSuite, RefinementWrapperSuite, ScalacheckSuite, should.Matchers:
    test("Unsafe wrapper works; frame index must be nonnegative.") {
        forAll { (z: Int) => 
            if z < 0 
            then assertThrows[NumberFormatException]{ Timepoint.unsafe(z) }
            else Timepoint.unsafe(z).get shouldEqual z
        }
    }

    test("Frame indices are equivalent on their wrapped values.") {
        forAll (genEquivalenceInputAndExpectation(Timepoint.apply)) { case (f1, f2, exp) => f1 === f2 shouldBe exp }
    }

    test("Set respects frame index equivalence.") {
        forAll (genValuesAndNumUnique(Gen.choose(0, 100))(Timepoint.unsafe)) { 
            case (indices, expected) => indices.toSet shouldEqual expected
        }
    }
end TestTimepoint
