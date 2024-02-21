package at.ac.oeaw.imba.gerlich.looptrace

import cats.data.{ NonEmptyList, NonEmptySet }
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.funsuite.AnyFunSuite

/**
  * Tests for [[at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundConfiguration]]
  * 
  * @author Vince Reuter
  */
class TestImagingRoundConfiguration extends AnyFunSuite, GenericSuite, ScalacheckSuite, should.Matchers:
    test("Example config parses correctly.") {
        exampleConfig.numberOfRounds shouldEqual 11
        exampleConfig.regionalGrouping shouldEqual ImagingRoundConfiguration.RegionalGrouping.Permissive(
            NonEmptyList.of(NonEmptySet.of(7, 8), NonEmptySet.of(9, 10)).map(_.map(Timepoint.unsafe))
        )
        exampleConfig.tracingExclusions shouldEqual Set(0, 7, 8, 9, 10).map(Timepoint.unsafe)
        val (blankRounds, locusRounds, regionalRounds) = exampleConfig.sequenceOfRounds.rounds.toList.foldRight(
            (List.empty[BlankImagingRound], List.empty[LocusImagingRound], List.empty[RegionalImagingRound])
            ){ 
                case (r: BlankImagingRound, (blanks, locals, regionals)) => (r :: blanks, locals, regionals)
                case (r: LocusImagingRound, (blanks, locals, regionals)) => (blanks, r :: locals, regionals)
                case (r: RegionalImagingRound, (blanks, locals, regionals)) => (blanks, locals, r :: regionals)
            }
        blankRounds.map(_.name) shouldEqual List("pre_image", "blank_01")
        locusRounds.map(_.name) shouldEqual locusRounds.map(_.probe.get)  // Name inferred from probe when not explicit
        locusRounds.map(_.probe) shouldEqual List("Dp001", "Dp002", "Dp003", "Dp006", "Dp007").map(ProbeName.apply)
        regionalRounds.map(_.name) shouldEqual regionalRounds.map(_.probe.get)
        regionalRounds.map(_.probe) shouldEqual List("Dp101", "Dp102", "Dp103", "Dp104").map(ProbeName.apply)
    }

    private lazy val exampleConfig: ImagingRoundConfiguration = {
        val configFile = getResourcePath("example_imaging_round_configuration.json")
        ImagingRoundConfiguration.unsafeFromJsonFile(configFile)
    }
    
    private def getResourcePath(name: String): os.Path = 
        os.Path(getClass.getResource(s"/TestImagingRoundConfiguration/$name").getPath)
end TestImagingRoundConfiguration
