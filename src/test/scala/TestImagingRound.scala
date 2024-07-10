package at.ac.oeaw.imba.gerlich.looptrace

import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

/**
  * Tests for the abstraction of an imaging round
  * 
  * @author Vince Reuter
  */
class TestImagingRound extends AnyFunSuite, ScalaCheckPropertyChecks, ImagingRoundHelpers, LooptraceSuite, should.Matchers:
    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)

    test("ImagingRound itself cannot be instantiated.") {
        assertTypeError{ "new ImagingRound{ def name = \"absolutelynot\"; def timepoint = ImagingTimepoint(NonnegativeInt(0)) }" }
    }
    
    test("BlankImagingRound can be instantiated (DIRECTLY) with just name and timepoint, but must have appropriate key-value to come from JSON through parent parser.") {
        assertCompiles{ "BlankImagingRound(\"absolutelynot\", ImagingTimepoint(NonnegativeInt(0)))" }
        forAll { (name: String, time: ImagingTimepoint) => 
            val baseData = Map("name" -> ujson.Str(name), "time" -> ujson.Num(time.get))
            ImagingRound.parseFromJsonMap(baseData).isLeft shouldBe true
            ImagingRound.parseFromJsonMap(baseData + ("isBlank" -> ujson.Bool(false))).isLeft shouldBe true
            ImagingRound.parseFromJsonMap(baseData + ("isBlank" -> ujson.Bool(true))) shouldEqual BlankImagingRound(name, time).asRight
        }
    }

    test("Non-blank round without probe contains appropriate error message.") {
        forAll { (name: String, time: ImagingTimepoint, useBlankKey: Boolean, regOpt: Option[Boolean], repOpt: Option[PositiveInt]) => 
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

    test("Blank round is parsed from name + time + blank flag.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        forAll (genNameForJson, arbitrary[ImagingTimepoint]) { (name, time) => 
            val data = s"""{\"name\": \"$name\", \"time\": ${time.show_}, \"isBlank\": true}"""
            given reader: Reader[ImagingRound] = ImagingRound.rwForImagingRound
            read[ImagingRound](data) shouldEqual BlankImagingRound(name, time)
        }
    }

    test("With probe and no blank flag (or blank flag set to false), region flag dictates type parsed.") {
        given arbName: Arbitrary[String] = genNameForJson.toArbitrary
        given reader: Reader[ImagingRound] = ImagingRound.rwForImagingRound
        forAll { (optName: Option[String], time: ImagingTimepoint, probe: ProbeName, specifyBlankIsFalse: Boolean, optRegion: Option[Boolean]) =>
            val baseData = List("time" -> ujson.Num(time.get), "probe" -> ujson.Str(probe.get))
            val extraData = List(
                optName.map(n => "name" -> ujson.Str(n)),
                specifyBlankIsFalse.option("isBlank" -> ujson.Bool(false)), 
                optRegion.map(p => "isRegional" -> ujson.Bool(p)), 
                ).flatten
            val jsonText = write((baseData ++ extraData).toMap)
            val expect = if optRegion.getOrElse(false) then RegionalImagingRound(optName, time, probe) else LocusImagingRound(optName, time, probe, None)
            read[ImagingRound](jsonText) shouldEqual expect
        }
    }

    test("BlankImagingRound can roundtrip through JSON.") {
        given rw: (Reader[BlankImagingRound] & Writer[BlankImagingRound]) = ImagingRound.rwForImagingRound.narrow[BlankImagingRound]
        forAll { (blank: BlankImagingRound) => blank shouldEqual read[BlankImagingRound](write(blank)) }
    }

    test("BlankImagingRound cannot have a probe.") {
        given rw: (Reader[BlankImagingRound] & Writer[BlankImagingRound]) = ImagingRound.rwForImagingRound.narrow[BlankImagingRound]

        assertTypeError{ "BlankImagingRound(\"absolutelynot\", ImagingTimepoint(NonnegativeInt(0)), ProbeName(\"irrelevant\"))" }
        forAll { (blank: BlankImagingRound, probe: ProbeName) => 
            val data = ImagingRound.roundToJsonObject(blank).obj.toMap + ("probe" -> ujson.Str(probe.get))
            val error = intercept[ImagingRound.DecodingError]{ read[BlankImagingRound](write(data)) }
            error.whatWasBeingDecoded shouldEqual "ImagingRound"
            error.messages.toList.count(_ ===  "Blank frame cannot have probe specified!") shouldEqual 1
        }
    }

    test("Blank vs. locus-specific: either probe or blank flag is required.") {
        given arbName: Arbitrary[String] = genNameForJson.toArbitrary
        given reader: Reader[ImagingRound] = ImagingRound.rwForImagingRound
        
        /* Avoid passing a repeat value for a blank round. */
        def genBlankAndRepeat = Gen.zip(
            Gen.option(Gen.oneOf(false, true)), 
            arbitrary[Option[PositiveInt]]
            ).suchThat((optBlank, optRep) => !(optBlank.getOrElse(false) && optRep.nonEmpty))
        
        forAll (arbitrary[String], arbitrary[ImagingTimepoint], arbitrary[Boolean], arbitrary[Option[ProbeName]], genBlankAndRepeat) { 
            case (name, time, explicitlyNonRegional, optProbe, (optBlank, optRepeat)) =>
                val baseData = List("name" -> ujson.Str(name), "time" -> ujson.Num(time.get))
                val extraData = List(
                    optBlank.map{ p => "isBlank" -> ujson.Bool(p) },
                    optProbe.map{ p => "probe" -> ujson.Str(p.get) },
                    explicitlyNonRegional.option{ "isRegional" -> ujson.Bool(false) }, 
                    optRepeat.map{ n => "repeat" -> ujson.Num(n) },
                    ).flatten
                val jsonText = write((baseData ::: extraData).toMap)
                (optBlank, optProbe) match {
                    case (None | Some(false), None) => 
                        val error = intercept[ImagingRound.DecodingError]{ read[ImagingRound](jsonText) }
                        error.messages.toList.count(_ === "Probe is required when a round isn't blank!") shouldEqual 1
                    case (None | Some(false), Some(probe)) => read[ImagingRound](jsonText) shouldEqual LocusImagingRound(name.some, time, probe, optRepeat)
                    case (Some(true), None) => read[ImagingRound](jsonText) shouldEqual BlankImagingRound(name, time)
                    case (Some(true), Some(_)) => 
                        val error = intercept[ImagingRound.DecodingError]{ read[ImagingRound](jsonText) }
                        error.messages.toList.count(_ === "Blank frame cannot have probe specified!") shouldEqual 1
                }
        }
    }

    test("Repeat index is correctly added to locus round name...if and only if name isn't explicitly provided.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        given arbName: Arbitrary[String] = genNameForJson.toArbitrary
        given reader: Reader[ImagingRound] = ImagingRound.rwForImagingRound
        
        forAll { (optName: Option[String], time: ImagingTimepoint, probe: ProbeName, explicitlyNonBlank: Boolean, explicitlyNonRegional: Boolean, optRepeat: Option[PositiveInt]) => 
            val baseData = List("time" -> ujson.Num(time.get), "probe" -> ujson.Str(probe.get))
            val extraData = List(
                optName.map{ n => "name" -> ujson.Str(n) },
                explicitlyNonBlank.option{ "isBlank" -> ujson.Bool(false) }, 
                explicitlyNonRegional.option{ "isRegional" -> ujson.Bool(false) },
                optRepeat.map{ n => "repeat" -> ujson.Num(n) },
                ).flatten
            val jsonText = write((baseData ::: extraData).toMap)
            val expect = optName.getOrElse{ probe.get ++ optRepeat.fold("")(n => s"_repeat${n.show_}") }
            read[ImagingRound](jsonText) match {
                case round: LocusImagingRound => round.name shouldEqual expect
                case round => fail(s"Expected a locus imaging round but got $round")
            }
        }
    }

    test("Specifying that a blank frame is a repeat is an error, since doing so would have no effect.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        given arbName: Arbitrary[String] = genNameForJson.toArbitrary
        given reader: Reader[ImagingRound] = ImagingRound.rwForImagingRound

        forAll { (name: String, time: ImagingTimepoint, repeat: PositiveInt) => 
            val data = Map("name" -> ujson.Str(name), "time" -> ujson.Num(time.get), "isBlank" -> ujson.Bool(true), "repeat" -> ujson.Num(repeat))
            val jsonText = write(data)
            val error = intercept[ImagingRound.DecodingError]{ read[ImagingRound](jsonText) }
            error.messages.toList.count(_ === "Blank round cannot be a repeat!") shouldEqual 1
        }
    }
end TestImagingRound
