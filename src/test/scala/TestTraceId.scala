package at.ac.oeaw.imba.gerlich.looptrace

import cats.syntax.eq.*
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given

/** Tests for trace ID wrapper type */
class TestTraceId extends AnyFunSuite, GenericSuite, ScalaCheckPropertyChecks, RefinementWrapperSuite, should.Matchers:
    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)

    test("TraceId.fromInt works; trace ID must be nonnegative.") {
        forAll { (z: Int) => TraceId.fromInt(z) match {
            case Left(msg) if z < 0 => msg shouldEqual s"Cannot refine as nonnegative: $z"
            case Right(tid) if z >= 0 => tid.get shouldEqual z
            case result => fail(s"Unexpected result parsing trace ID from int ($z): $result")
        } }
    }

    test("Trace IDs are equivalent on their wrapped values.") {
        forAll (genEquivalenceInputAndExpectation(TraceId.apply)) { case (f1, f2, exp) => f1 === f2 shouldBe exp }
    }

    test("Set respects trace ID equivalence.") {
        forAll (genValuesAndNumUnique(arbitrary[NonnegativeInt])(TraceId.apply)) { 
            case (ids, exp) => ids.toSet shouldEqual exp
        }
    }
end TestTraceId
