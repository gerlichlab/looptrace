package at.ac.oeaw.imba.gerlich.looptrace

import org.scalacheck.Gen
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should

/** Tests for positive integer refinement type */
class TestPositiveInt extends AnyFunSuite, ScalacheckSuite, should.Matchers:
    test("PositiveInt correctly restricts which expressions compile.") {
        assertCompiles("PositiveInt(1)")
        assertTypeError("PositiveInt(0)")
        assertTypeError("PositiveInt(-1)")
    }

    test("PositiveInt.maybe behaves correctly.") {
        forAll { (z: Int) => PositiveInt.maybe(z) match {
            case None if z <= 0 => succeed
            case Some(n) if z > 0 => z shouldEqual n
            case bad => fail(s"PositiveInt.maybe($z) gave bad result: $bad")
        } }
    }

    test("PositiveInt.unsafe behaves in accordance with its safe counterpart.") {
        forAll { (z: Int) => PositiveInt.maybe(z) match {
            case None => assertThrows[NumberFormatException]{ PositiveInt.unsafe(z) }
            case Some(n) => n shouldEqual PositiveInt.unsafe(z)
        } }
    }

    test("Natural numbers are a subset of nonnegative integers.") {
        forAll(Gen.posNum[Int]) {
            n => PositiveInt.unsafe(n).asNonnegative shouldEqual NonnegativeInt.unsafe(n)
        }
    }
end TestPositiveInt
