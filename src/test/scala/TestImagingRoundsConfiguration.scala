package at.ac.oeaw.imba.gerlich.looptrace

import cats.data.{ NonEmptyList, NonEmptySet }

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.LocusGroup

/**
  * Tests for [[at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration]]
  * 
  * @author Vince Reuter
  */
class TestImagingRoundsConfiguration extends AnyFunSuite, LooptraceSuite, ScalaCheckPropertyChecks, should.Matchers:
    implicit override val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)

    test("Example config parses correctly.") {
        exampleConfig.numberOfRounds shouldEqual 12
        exampleConfig.regionGrouping shouldEqual ImagingRoundsConfiguration.RegionGrouping.Permissive(
            NonEmptyList.of(NonEmptySet.of(8, 9), NonEmptySet.of(10, 11)).map(_.map(Timepoint.unsafe))
        )
        exampleConfig.tracingExclusions shouldEqual Set(0, 8, 9, 10, 11).map(Timepoint.unsafe)
        val (blankRounds, locusRounds, regionalRounds) = exampleConfig.sequenceOfRounds.rounds.toList.foldRight(
            (List.empty[BlankImagingRound], List.empty[LocusImagingRound], List.empty[RegionalImagingRound])
            ){ 
                case (r: BlankImagingRound, (blanks, locals, regionals)) => (r :: blanks, locals, regionals)
                case (r: LocusImagingRound, (blanks, locals, regionals)) => (blanks, r :: locals, regionals)
                case (r: RegionalImagingRound, (blanks, locals, regionals)) => (blanks, locals, r :: regionals)
                case _ => 
                    // Impossible, only here b/c as of 2024-02-21, on Scala 3.3.2 this wasn't compiling
                    // In particular, there was a match exhaustivity error seemingly related to the following: 
                    // https://github.com/scala/bug/issues/9677
                    ???
            }
        blankRounds.map(_.name) shouldEqual List("pre_image", "blank_01")
        locusRounds.map(_.name).init shouldEqual locusRounds.map(_.probe.get).init  // Name inferred from probe when not explicit
        locusRounds.last.name shouldEqual locusRounds.last.probe.get ++ "_repeat1"
        locusRounds.map(_.probe) shouldEqual List("Dp001", "Dp002", "Dp003", "Dp006", "Dp007", "Dp001").map(ProbeName.apply)
        regionalRounds.map(_.name) shouldEqual regionalRounds.map(_.probe.get)
        regionalRounds.map(_.probe) shouldEqual List("Dp101", "Dp102", "Dp103", "Dp104").map(ProbeName.apply)
        exampleConfig.locusGrouping shouldEqual NonEmptySet.of(
            8 -> NonEmptySet.of(1, 6),
            9 -> NonEmptySet.one(2), 
            10 -> NonEmptySet.of(3, 4),
            11 -> NonEmptySet.one(5)
            )
            .map{ (r, ls) => Timepoint.unsafe(r) -> ls.map(Timepoint.unsafe) }
            .map(ImagingRoundsConfiguration.LocusGroup.apply.tupled)
    }

    test("Region grouping must either be entirely absent or must specify a valid semantic.") { pending }

    test("Region grouping must be either entirely absent or specify groups that constitute a partition of regional round timepoints from the imaging sequence.") { pending }

    test("Locus grouping must be present and have a collection of values that constitutes a partition of locus imaging rounds from the imaging sequence.") { pending }

    test("Each of the locus grouping's keys must be a regional round timepoint from the imaging sequence") { pending }

    test("Any timepoint to exclude from tracing must be a timepoint in the imaging sequence.") { pending }

    test("Configuration IS allowed to have regional rounds in sequence that have no loci in locus grouping, #270.") { pending }
    
    private lazy val exampleConfig: ImagingRoundsConfiguration = {
        val configFile = getResourcePath("example_imaging_round_configuration.json")
        ImagingRoundsConfiguration.unsafeFromJsonFile(configFile)
    }
    
    private def getResourcePath(name: String): os.Path = 
        os.Path(getClass.getResource(s"/TestImagingRoundsConfiguration/$name").getPath)


    private def genRound(using arbName: Arbitrary[String], arbTime: Arbitrary[Timepoint]): Gen[ImagingRound] = Gen.oneOf(
        arbitrary[BlankImagingRound], 
        arbitrary[RegionalImagingRound], 
        arbitrary[LocusImagingRound],
        )

    private def namesAreUnique(rounds: List[ImagingRound]): Boolean = rounds.map(_.name).toSet.size === rounds.length
end TestImagingRoundsConfiguration
