package at.ac.oeaw.imba.gerlich.looptrace

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

/** Tests for the filtration of FISH spots/ROIs through nuclei */
class TestFilterSpotsByProximity
    extends AnyFunSuite,
      LooptraceSuite,
      ScalaCheckPropertyChecks,
      should.Matchers:
  test("Proximity-based filtration alters no records."):
    pending

  test("Proximity-based filtration can't increase record count."):
    pending
end TestFilterSpotsByProximity
