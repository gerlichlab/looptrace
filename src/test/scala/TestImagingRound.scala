package at.ac.oeaw.imba.gerlich.looptrace

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*

/**
  * Tests for the abstraction of an imaging round
  * 
  * @author Vince Reuter
  */
class TestImagingRound extends AnyFunSuite, DistanceSuite, LooptraceSuite, ScalacheckSuite, should.Matchers:
    test("ImagingRound itself cannot be instantiated.") { pending }
    test("FishImagingRound itself cannot be instantiated.") { pending }
    test("BlankImagingRound takes just name and timepoint.") { pending }
end TestImagingRound
