package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Random
import cats.data.{ NonEmptyList, NonEmptySet }
import cats.syntax.all.*

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
        val seq = exampleConfig.sequence
        seq.blankRounds.map(_.name) shouldEqual List("pre_image", "blank_01")
        seq.locusRounds.map(_.name).init shouldEqual seq.locusRounds.map(_.probe.get).init  // Name inferred from probe when not explicit
        seq.locusRounds.last.name shouldEqual seq.locusRounds.last.probe.get ++ "_repeat1"
        seq.locusRounds.map(_.probe) shouldEqual List("Dp001", "Dp002", "Dp003", "Dp006", "Dp007", "Dp001").map(ProbeName.apply)
        seq.regionRounds.map(_.name) shouldEqual seq.regionRounds.map(_.probe.get)
        seq.regionRounds.map(_.probe) shouldEqual NonEmptyList.of("Dp101", "Dp102", "Dp103", "Dp104").map(ProbeName.apply)
        exampleConfig.locusGrouping shouldEqual NonEmptySet.of(
            8 -> NonEmptySet.of(1, 6),
            9 -> NonEmptySet.one(2), 
            10 -> NonEmptySet.of(3, 4),
            11 -> NonEmptySet.one(5)
            )
            .map{ (r, ls) => Timepoint.unsafe(r) -> ls.map(Timepoint.unsafe) }
            .map(ImagingRoundsConfiguration.LocusGroup.apply.tupled)
    }

    // test("Region grouping must either be trivial or must specify a valid semantic.") {
    //     def mygen = for {
    //         seq <- genTimeValidSequence(Gen.choose(1, 10).map(PositiveInt.unsafe))
    //         regionGrouping <- Gen.option(genValidParitionForRegionGrouping(seq)).flatMap{
    //             case None => Gen.const(ImagingRoundsConfiguration.RegionGrouping.Trivial)
    //             case Some(parts) => 
    //                 arbitrary[ImagingRoundsConfiguration.RegionGrouping.Semantic].map{
    //                     semantic => ImagingRoundsConfiguration.RegionGrouping.Nontrivial(_, parts)
    //                 }
    //         }
    //         locusGrouping <- 
    //         exclusions <- genExclusions(seq)
    //     } yield (seq, regionGrouping, exclusions)
    // }

    test("Region grouping must either be trivial or must specify groups that constitute a partition of regional round timepoints from the imaging sequence.") { pending }

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

    private def genExclusions(sequence: ImagingSequence): Gen[Set[Timepoint]] = 
        Gen.choose(0, sequence.length)
            .flatMap(Gen.pick(_, sequence.allTimepoints.toList))
            .map(_.toSet)

    private def genValidLocusGrouping(sequence: ImagingSequence): Gen[NonEmptyList[LocusGroup]] = ???

    private def genValidParitionForRegionGrouping(sequence: ImagingSequence): Gen[NonEmptyList[NonEmptySet[Timepoint]]] = {
        for {
            k <- Gen.choose(1, sequence.regionRounds.length)
            (g1, g2) = Random.shuffle(sequence.regionRounds.toList).toList.splitAt(k)
        } yield List(g1, g2).flatMap(_.toNel).map(_.map(_.time).toNes).toNel.get
    }

    private def genTimeValidSequence(genSize: Gen[PositiveInt])(using arbName: Arbitrary[String]): Gen[ImagingSequence] = 
        genSize.flatMap(k => (0 until k).toList.traverse{ t => 
            given arbT: Arbitrary[Timepoint] = Gen.const(Timepoint.unsafe(t)).toArbitrary
            genRound
        })
        .map(ImagingSequence.fromRounds)
        .suchThat(_.isRight)
        .map(_.getOrElse{ throw new Exception("Generated illegal imaging sequence despite controlling times and names!") })

    private def genRound(using arbName: Arbitrary[String], arbTime: Arbitrary[Timepoint]): Gen[ImagingRound] = Gen.oneOf(
        arbitrary[BlankImagingRound], 
        arbitrary[RegionalImagingRound], 
        arbitrary[LocusImagingRound],
        )
end TestImagingRoundsConfiguration
