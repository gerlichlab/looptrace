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
    
    test("ImagingRound itself cannot be instantiated.") {
        assertTypeError{ "new ImagingRound{ def name = \"absolutelynot\"; def timepoint = Timepoint(NonnegativeInt(0)) }" }
    }
    
    test("FishImagingRound itself cannot be instantiated.") {
        assertTypeError{ "new FishImagingRound{ def name = \"absolutelynot\"; def timepoint = Timepoint(NonnegativeInt(0)); def probe = ProbeName(\"irrelevant\"); def repeat = Option.empty[PositiveInt] }" }
    }
    
    test("BlankImagingRound takes just name and timepoint.") {
        assertCompiles{ "BlankImagingRound(\"absolutelynot\", Timepoint(NonnegativeInt(0)))" }
    }
end TestImagingRound
