package at.ac.oeaw.imba.gerlich.looptrace

import cats.data.{ NonEmptyList, NonEmptySet }
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.funsuite.AnyFunSuite
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundConfiguration.LocusGroup

/**
  * Tests for [[at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundConfiguration]]
  * 
  * @author Vince Reuter
  */
class TestImagingRoundConfiguration extends AnyFunSuite, GenericSuite, ScalacheckSuite, should.Matchers:
    test("Example config parses correctly.") {
        exampleConfig.numberOfRounds shouldEqual 12
        exampleConfig.regionGrouping shouldEqual ImagingRoundConfiguration.RegionGrouping.Permissive(
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
            .map(ImagingRoundConfiguration.LocusGroup.apply.tupled)

    }

    private lazy val exampleConfig: ImagingRoundConfiguration = {
        val configFile = getResourcePath("example_imaging_round_configuration.json")
        ImagingRoundConfiguration.unsafeFromJsonFile(configFile)
    }
    
    private def getResourcePath(name: String): os.Path = 
        os.Path(getClass.getResource(s"/TestImagingRoundConfiguration/$name").getPath)
end TestImagingRoundConfiguration
