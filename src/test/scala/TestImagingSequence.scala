package at.ac.oeaw.imba.gerlich.looptrace

import cats.Order
import cats.syntax.all.*
import upickle.default.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/**
  * Tests for config file definition and parsing of imaging rounds and a sequence of them for an experiment.
  * 
  *  @author Vince Reuter
  */
class TestImagingSequence extends AnyFunSuite, ScalaCheckPropertyChecks, ImagingRoundHelpers, LooptraceSuite, should.Matchers:
    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)
    
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
    
    test("Sequence of timepoints other than 0, 1, ..., N-1 gives expected error.") {
        given [A] => Shrink[A] = Shrink.shrinkAny[A]
        given arbName: Arbitrary[String] = genNameForJson.toArbitrary
        def genRoundsSeq: Gen[List[ImagingRound]] = 
            given Ordering[ImagingTimepoint] = summon[Order[ImagingTimepoint]].toOrdering // to check sorted list
            Gen.nonEmptyListOf(genRound)
                .suchThat(namesAreUnique)
                .suchThat(rs => rs.map(_.time).sorted =!= (0 until rs.length).toList.map(ImagingTimepoint.unsafe))
        forAll (genRoundsSeq, minSuccessful(1000)) { 
            (rounds: List[ImagingRound]) => 
                ImagingSequence.fromRounds(rounds) match {
                    case Left(messages) => 
                        val expPrefix = "Ordered timepoints for imaging rounds don't form contiguous sequence"
                        val obsCount = messages.toList.count(_.startsWith(expPrefix))
                        if 1 === obsCount
                        then succeed
                        else
                            val errMsg = s"Expected exactly 1 message prefixed by '$expPrefix', but got $obsCount"
                            println(errMsg)
                            println("Here are the messages (below):")
                            messages.toList.foreach(println)
                            fail(errMsg)
                    case Right(_) => fail("Expected ImagingSequence parse error(s) but got success!")
                }
        }
    }
    
    test("Non-unique names is an error.") {
        given [A] => Shrink[A] = Shrink.shrinkAny[A]
        def genRoundsWithNameCollision = {
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
    
    test("List of imaging round declarations can roundtrip through JSON.") {
        given [A] => Shrink[A] = Shrink.shrinkAny[A]
        given rw: ReadWriter[ImagingRound] = ImagingRound.rwForImagingRound
        def genRounds = Gen.choose(1, 100)
            /* Force satisfaction of the requirement that the timepoints form sequence [0, ..., N-1]. */
            .map(k => (0 until k).map(ImagingTimepoint.unsafe).toList)
            .flatMap(_.traverse{ t => genRound(using genNameForJson.toArbitrary, Gen.const(t).toArbitrary) })
            .suchThat(namesAreUnique)
        forAll (genRounds) { rounds => 
            val exp = ImagingSequence.fromRounds(rounds)
            // Write the rounds to JSON, and then read/parse them back, checking that the result matches the original.
            val obs = ImagingSequence.fromRounds(read[List[ImagingRound]](write(rounds)))
            obs shouldEqual exp
        }
    }

    test("Through JSON is identical to directly from rounds.") {
        given [A] => Shrink[A] = Shrink.shrinkAny[A]
        given arbName: Arbitrary[String] = genNameForJson.toArbitrary
        given rw: ReadWriter[ImagingRound] = ImagingRound.rwForImagingRound
        pending
    }

    private def genRound(using arbName: Arbitrary[String], arbTime: Arbitrary[ImagingTimepoint]): Gen[ImagingRound] = Gen.oneOf(
        arbitrary[BlankImagingRound], 
        arbitrary[RegionalImagingRound], 
        arbitrary[LocusImagingRound],
    )

    private def namesAreUnique(rounds: List[ImagingRound]): Boolean = rounds.map(_.name).toSet.size === rounds.length
end TestImagingSequence
