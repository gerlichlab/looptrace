package at.ac.oeaw.imba.gerlich.looptrace

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*

import upickle.default.*

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
    
    test("BlankImagingRound can be instantiated with just name and timepoint.") {
        assertCompiles{ "BlankImagingRound(\"absolutelynot\", Timepoint(NonnegativeInt(0)))" }
    }

    test("Just name and timepoint is under-specified (blank or probe is required.)") { pending }

    test("BlankImagingRound can roundtrip through JSON.") {
        forAll { (blank: BlankImagingRound) => 
            given rw: (Reader[BlankImagingRound] & Writer[BlankImagingRound]) = ImagingRound.rwForImagingRound.narrow[BlankImagingRound]
            blank shouldEqual read[BlankImagingRound](write(blank))
        }
    }

    test("BlankImagingRound cannot have a probe.") {
        assertTypeError{ "BlankImagingRound(\"absolutelynot\", Timepoint(NonnegativeInt(0)), ProbeName(\"irrelevant\"))" }
        // val error = intercept[ImagingRound.DecodingError]{ ImagingRound.parseFromJsonMap(baseData + ("probe" -> ujson.Str(probe.get))) }
        //     error.whatWasBeingDecoded shouldEqual "ImagingRound"
        //     error.errors.toList.count(_ ===  "Blank frame cannot have probe specified!") shouldEqual 1
    }

    test("Non-blank round must have probe.") { pending }

    test("Presence of probe makes distinguishes blank round from locus-specific round.") { pending }

    test("Presence of regional flag distinguishes regional from locus-specific.") { pending }
end TestImagingRound
