package at.ac.oeaw.imba.gerlich.looptrace

import mouse.boolean.*
import upickle.default.*
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
    
    test("BlankImagingRound can be instantiated with just name and timepoint.") {
        assertCompiles{ "BlankImagingRound(\"absolutelynot\", Timepoint(NonnegativeInt(0)))" }
    }

    test("Non-blank round without probe contains appropriate error message.") {
        forAll { (name: String, time: Timepoint, useBlankKey: Boolean, regOpt: Option[Boolean], repOpt: Option[PositiveInt]) => 
            val baseData = Map("name" -> ujson.Str(name), "time" -> ujson.Num(time.get))
            val extras = List(
                regOpt.map{ p => "isRegional" -> ujson.Bool(p) }, 
                repOpt.map{ n => "repeat" -> ujson.Num(n) }, 
                useBlankKey.option{ "isBlank" -> ujson.Bool(false) }
                ).flatten.toMap
            ImagingRound.parseFromJsonMap(baseData ++ extras) match {
                case Left(messages) => messages.toList.count(_ ===  "Probe is required when a round isn't blank!") shouldEqual 1
                case Right(_) => fail("Expected decoding failure, but it succeded!")
            }
        }
    }

    test("Blank round is parsed from name + time + blank flag.") { pending }

    test("Locus round is parsed from JUST probe + time.") { pending }
        
    test("Locus or region is parsed from probe + time + regional flag + (optional) name.") { pending }

    test("BlankImagingRound can roundtrip through JSON.") {
        given rw: (Reader[BlankImagingRound] & Writer[BlankImagingRound]) = ImagingRound.rwForImagingRound.narrow[BlankImagingRound]
        forAll { (blank: BlankImagingRound) => blank shouldEqual read[BlankImagingRound](write(blank)) }
    }

    test("BlankImagingRound cannot have a probe.") {
        given rw: (Reader[BlankImagingRound] & Writer[BlankImagingRound]) = ImagingRound.rwForImagingRound.narrow[BlankImagingRound]
        assertTypeError{ "BlankImagingRound(\"absolutelynot\", Timepoint(NonnegativeInt(0)), ProbeName(\"irrelevant\"))" }
        forAll { (blank: BlankImagingRound, probe: ProbeName) => 
            val data = ImagingRound.roundToJsonObject(blank).obj.toMap + ("probe" -> ujson.Str(probe.get))
            val error = intercept[ImagingRound.DecodingError]{ read[BlankImagingRound](write(data)) }
            error.whatWasBeingDecoded shouldEqual "ImagingRound"
            error.messages.toList.count(_ ===  "Blank frame cannot have probe specified!") shouldEqual 1
        }
    }

    test("Presence of probe makes distinguishes blank round from locus-specific round.") { pending }

    test("Presence of regional flag distinguishes regional from locus-specific.") { pending }
end TestImagingRound
