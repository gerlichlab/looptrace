package at.ac.oeaw.imba.gerlich.looptrace

import cats.syntax.eq.*
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

/** Tests for position index wrapper type */
class TestProbeName extends AnyFunSuite, ScalaCheckPropertyChecks, RefinementWrapperSuite, should.Matchers:
    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)
    
    test("Probe names are equivalent on their wrapped values.") {
        forAll (genEquivalenceInputAndExpectation(ProbeName.apply)) { case (f1, f2, exp) => f1 === f2 shouldBe exp }
    }

    test("Set respects probe name equivalence.") {
        forAll (genValuesAndNumUnique(arbitrary[String])(ProbeName.apply)) {
            case (indices, expected) => indices.toSet shouldEqual expected
        }
    }
end TestProbeName
