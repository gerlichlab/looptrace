package at.ac.oeaw.imba.gerlich.looptrace

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*

/** Tests for the filtration of the individual supports (single FISH probes) of chromatin fiber traces */
class TestLabelAndFilterRois extends AnyFunSuite, GenericSuite, ScalacheckSuite, ScalacheckGenericExtras, should.Matchers:
    test("Spot distance comparison requires drift correction") { pending }
    test("Spot distance comparison uses drift correction.") { pending }
    test("Spot distance comparison responds to change of proximity comparison strategy.") { pending }
    test("In each pair of proximal spots, BOTH are filtered. #148") { pending }
end TestLabelAndFilterRois
