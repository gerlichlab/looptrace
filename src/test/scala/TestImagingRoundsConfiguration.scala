package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Random
import cats.data.{ NonEmptyList, NonEmptySet }
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.{ LocusGroup, RegionGrouping }
import at.ac.oeaw.imba.gerlich.looptrace.space.*

/**
  * Tests for [[at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration]]
  * 
  * @author Vince Reuter
  */
class TestImagingRoundsConfiguration extends AnyFunSuite, LooptraceSuite, ScalaCheckPropertyChecks, should.Matchers:
    implicit override val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)

    test("Example config parses correctly.") {
        val exampleConfig: ImagingRoundsConfiguration = {
            val configFile = getResourcePath("example_imaging_round_configuration.json")
            ImagingRoundsConfiguration.unsafeFromJsonFile(configFile)
        }
        exampleConfig.numberOfRounds shouldEqual 12
        exampleConfig.regionGrouping shouldEqual RegionGrouping.Permissive(
            PiecewiseDistance.ConjunctiveThreshold(NonnegativeReal(5.0)), 
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
            .map(LocusGroup.apply.tupled)
    }

    test("Missing region grouping gives the expected error.") {
        forAll (genValidSeqAndLocusGroupOptAndExclusions(PositiveInt(10))) { (seq, optLocusGrouping, exclusions) => 
            val baseData: Map[String, ujson.Value] = {
                val records: NonEmptyList[ujson.Obj] = Random.shuffle(seq.allRounds.map(ImagingRound.roundToJsonObject).toList).toList.toNel.get
                Map("imagingRounds" -> ujson.Arr(records.toList*))
            }
            val data: Map[String, ujson.Value] = addLocusGroupingAndExclusions(baseData, optLocusGrouping, exclusions)
            ImagingRoundsConfiguration.fromJsonMap(data) match {
                case Right(_) => fail(s"Expected parse failure on account of missing region grouping, but it succeeded!")
                case Left(messages) => messages.count(_ === "Missing regionGrouping section!") shouldEqual 1
            }
        }
    }

    test("Trivial region grouping with groups specified is an error.") { pending }

    test("Region grouping missing semantic gives error, regardless of presence of groups.") { pending }

    test("Region grouping with groups but invalid semantic gives expected error.") { pending }

    test("Without groups list, region grouping semantic must be trivial.") { pending }

    test("Without groups present, semantic must be trivial (or a variant thereof).") {
        pending
    }

    test("With groups present, regionGrouping section must specify a valid semantic.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        type BadSemanticValue = String // Give context-specific meaning to bare String.
        type RawThreshold = NonnegativeReal

        def mygen: Gen[(ImagingSequence, (BadSemanticValue, RawThreshold, NonEmptyList[NonEmptySet[Timepoint]]), Option[NonEmptyList[LocusGroup]], Set[Timepoint])] = for {
            (seq, locusGroupingOpt, exclusions) <- genValidSeqAndLocusGroupOptAndExclusions(PositiveInt(10))
            (semantic, grouping) <- genValidParitionForRegionGrouping(seq).flatMap{ grouping => 
                Gen.alphaNumStr
                    .suchThat{ s => ! Set("permissive", "prohibitive", "trivial").contains(s.toLowerCase) }
                    .map(_ -> grouping)
            }
            rawThreshold <- arbitrary[NonnegativeReal]
        } yield (seq, (semantic, rawThreshold, grouping), locusGroupingOpt, exclusions)

        forAll (mygen, minSuccessful(1000)) { case (seq, (semantic, rawThreshold, grouping), locusGroupingOpt, exclusions) => 
            val baseData: Map[String, ujson.Value] = {
                val records: NonEmptyList[ujson.Obj] = 
                    Random.shuffle(seq.allRounds.map(ImagingRound.roundToJsonObject).toList).toList.toNel.get
                Map(
                    "imagingRounds" -> ujson.Arr(records.toList*), 
                    "regionGrouping" -> ujson.Obj(
                        "semantic" -> ujson.Str(semantic),
                        "min_spot_dist" -> ujson.Num(rawThreshold),
                        "groups" -> ujson.Arr(grouping.toList.map(ts => ujson.Arr(ts.toList.map(t => ujson.Num(t.get))*))*)
                        )
                )
            }
            val data: Map[String, ujson.Value] = addLocusGroupingAndExclusions(baseData, locusGroupingOpt, exclusions)
            ImagingRoundsConfiguration.fromJsonMap(data) match {
                case Right(_) => fail(s"Expected parse failure based on semantic '$semantic', but it succeeded!")
                case Left(messages) => 
                    val expMsg = s"Illegal value for regional grouping semantic: $semantic"
                    val numMatchMessages = messages.count(_ === expMsg).toInt
                    if numMatchMessages === 1 then succeed
                    else {
                        println(s"DATA (below)\n${write(data, indent = 4)}")
                        fail(s"Expected exactly one message match but got $numMatchMessages. Data are above.")
                    }
            }
        }
    }

    test("Region grouping must either be trivial or must specify groups that constitute a partition of regional round timepoints from the imaging sequence.") {
        pending
    }

    test("Locus grouping must be present and have a collection of values that constitutes a partition of locus imaging rounds from the imaging sequence.") {
        pending
    }

    test("Each of the locus grouping's keys must be a regional round timepoint from the imaging sequence") { pending }

    test("Any timepoint to exclude from tracing must be a timepoint in the imaging sequence.") { pending }

    test("Configuration IS allowed to have regional rounds in sequence that have no loci in locus grouping, #270.") { pending }
    
    given rwForSeq: ReadWriter[ImagingSequence] = 
            ImagingRoundsConfiguration.rwForImagingSequence(using ImagingRound.rwForImagingRound)
    
    given rwForTime: ReadWriter[Timepoint] = readwriter[ujson.Value]
        .bimap(time => ujson.Num(time.get), json => Timepoint.unsafe(json.int))

            private def addLocusGroupingAndExclusions(baseData: Map[String, ujson.Value], optLocusGrouping: Option[NonEmptyList[LocusGroup]], exclusions: Set[Timepoint]): Map[String, ujson.Value] = {
        baseData ++ List(
            optLocusGrouping.map{ gs => 
                val data: NonEmptyList[(String, ujson.Value)] = gs.map{ g => 
                    g.regionalTimepoint.show -> ujson.Arr(g.locusTimepoints.toList.map(t => ujson.Num(t.get))*)
                }
                "locusGrouping" -> ujson.Obj(data.head, data.tail*)
            }: Option[(String, ujson.Value)],
            exclusions.nonEmpty.option{
                "tracingExclusions" -> 
                ujson.Arr(exclusions.toList.map(t => ujson.Num(t.get))*)
            }: Option[(String, ujson.Value)]
        ).flatten
    }

    def genValidSeqAndLocusGroupOptAndExclusions(maxNumRounds: PositiveInt): Gen[(ImagingSequence, Option[NonEmptyList[LocusGroup]], Set[Timepoint])] = for {
        seq <- genTimeValidSequence(Gen.choose(1, 10).map(PositiveInt.unsafe))
        locusGroupingOpt: Option[NonEmptyList[LocusGroup]] <- seq.locusRounds.toNel match {
            case None => Gen.const(None)
            case Some(loci) => 
                if seq.regionRounds.length === PositiveInt(1) 
                then Gen.const(NonEmptyList.one(LocusGroup(seq.regionRounds.head.time, loci.map(_.time).toNes)).some)
                else {
                    val numLoci = seq.locusRounds.length
                    val numRegions: PositiveInt = seq.numberOfRegionRounds
                    (for {
                        // >= locus and >= 1 region, per the pattern match and conditional
                        k <- Gen.choose(1, math.min(numRegions, numLoci))
                        breaks <- Gen.pick(k - 1, 0 to numLoci).map(_.sorted :+ numLoci) // Ensure all loci are accounted for.
                    } yield breaks.zip(seq.regionRounds.toList).foldRight(loci.toList.map(_.time) -> List.empty[LocusGroup]){
                        case (_, (Nil, acc)) => Nil -> acc
                        case ((pt, reg), (remaining, acc)) => 
                            val (maybeNewGroup, newRemaining) = remaining.splitAt(pt)
                            val newAcc = maybeNewGroup.toNel.fold(acc)(ts => LocusGroup(reg.time, ts.toNes) :: acc)
                            (newRemaining, newAcc)
                    }._2.toNel)
                    .suchThat(_.nonEmpty)
                    .map(_.get.some)
                }
        }
        exclusions <- genExclusions(seq)
    } yield (seq, locusGroupingOpt, exclusions)

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
