package at.ac.oeaw.imba.gerlich.looptrace

import upickle.default.*
import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*

/**
  * Tests for config file definition and parsing of imaging rounds and a sequence of them for an experiment.
  * 
  *  @author Vince Reuter
  */
class TestImagingSequence extends AnyFunSuite, DistanceSuite, LooptraceSuite, ScalacheckSuite, should.Matchers:
    
    test("Empty collection is an error.") {
        ImagingSequence.fromRounds(List()) match {
            case Left(messages) => messages.toList match {
                case msg :: Nil => 
                    val exp = "Can't build an imaging sequence from empty collection of rounds!"
                    msg shouldEqual exp
                case _ => fail(s"Expected exactly 1 error message but got ${messages.length}!")
            }
            case Right(_) => fail("Expected imaging sequence parse to fail, but it succeeded!")
        }
    }
    
    test("Sequence of timepoints other than 0, 1, ..., N-1 is an error") { pending }
    
    test("Non-unique names is an error.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        def genRoundsWithNameCollision = {
            def genRound(using arbName: Arbitrary[String]): Gen[ImagingRound] = Gen.oneOf(
                arbitrary[BlankImagingRound], 
                arbitrary[RegionalImagingRound], 
                arbitrary[LocusImagingRound]
                )
            val namePool = List("firstName", "secondName")
            given arbName: Arbitrary[String] = Gen.oneOf(namePool).toArbitrary
            Gen.choose(namePool.size + 1, math.max(namePool.size + 1, 5)).flatMap(Gen.listOfN(_, genRound))
        }
        
        forAll (genRoundsWithNameCollision) { 
            rounds => ImagingSequence.fromRounds(rounds) match {
                case Left(messages) => 
                    messages.filter(_.startsWith("Repeated name(s) in imaging round sequence!")).length shouldEqual 1
                case Right(_) => fail("Expected imaging sequence parse to fail, but it succeeded!")
            }
        }
    }
    
    test("List of rounds can roundtrip through JSON.") { pending }
    
end TestImagingSequence
