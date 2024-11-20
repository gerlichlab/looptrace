package at.ac.oeaw.imba.gerlich.looptrace

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

/** Tests for the filtration of FISH spots/ROIs through nuclei */
class TestFilterSpotsThroughNuclei extends AnyFunSuite, LooptraceSuite, ScalaCheckPropertyChecks, should.Matchers:
    test("Nuclei-based filtration alters no records."):
        pending

    test("Nuclei-based filtration can't increase record count."):
        pending

    test("Nuclei-based filtration leaves every record with a strictly positive nucleus number/label."):
        pending
end TestFilterSpotsThroughNuclei
