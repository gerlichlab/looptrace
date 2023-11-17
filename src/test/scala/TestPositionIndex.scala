package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Random
import cats.syntax.eq.*
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should

/** Tests for position index wrapper type */
class TestPositionIndex extends AnyFunSuite, RefinementWrapperSuite, ScalacheckSuite, should.Matchers:
    test("Unsafe wrapper works; position index must be nonnegative.") {
        forAll { (z: Int) => 
            if z < 0 
            then assertThrows[NumberFormatException]{ PositionIndex.unsafe(z) }
            else PositionIndex.unsafe(z).get shouldEqual z
        }
    }

    test("Position indices are equivalent on their wrapped values.") {
        forAll (genEquivalenceInputAndExpectation(PositionIndex.apply)) { case (f1, f2, exp) => f1 === f2 shouldBe exp }
    }

    test("Set respects position index equivalence.") {
        forAll (genValuesAndNumUnique(PositionIndex.unsafe)) { case (indices, expected) => indices.toSet shouldEqual expected }
    }
end TestPositionIndex