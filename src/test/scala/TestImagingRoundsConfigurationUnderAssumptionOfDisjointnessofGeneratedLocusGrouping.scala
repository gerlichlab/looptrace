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

import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.{ LocusGroup, SelectiveProximityPermission }
import at.ac.oeaw.imba.gerlich.looptrace.space.*

/**
  * Tests for [[at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration]]
  * 
  * @author Vince Reuter
  */
class TestImagingRoundsConfigurationUnderAssumptionOfDisjointnessofGeneratedLocusGrouping extends AnyFunSuite, LooptraceSuite, ScalaCheckPropertyChecks, should.Matchers:
    implicit override val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)

    test("Example config parses correctly.") {
        val expectedNonemptyLocusGrouping = 
            NonEmptySet.of(
                8 -> NonEmptySet.of(1, 6),
                9 -> NonEmptySet.one(2), 
                10 -> NonEmptySet.of(3, 4),
                11 -> NonEmptySet.one(5)
            )
            .map{ (r, ls) => Timepoint.unsafe(r) -> ls.map(Timepoint.unsafe) }
            .map(LocusGroup.apply.tupled)
        forAll (Table(
            ("configFileName", "maybeExpectedLocusGrouping"), 
            ("example_imaging_round_configuration.json", expectedNonemptyLocusGrouping.some),
            ("rounds_config_with_empty_locus_grouping.json", None), 
            ("rounds_config_with_null_locus_grouping.json", None), 
            ("rounds_config_without_locus_grouping.json", None), 
        )) { (configFileName, maybeExpectedLocusGrouping) => 
            val exampleConfig: ImagingRoundsConfiguration = {
                val configFile = getResourcePath(configFileName)
                ImagingRoundsConfiguration.unsafeFromJsonFile(configFile)
            }
            exampleConfig.numberOfRounds shouldEqual 12
            exampleConfig.proximityFilterStrategy shouldEqual SelectiveProximityPermission(
                PositiveReal(5.0), 
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
            val expLocusGrouping: Set[LocusGroup] = maybeExpectedLocusGrouping.fold(Set.empty)(_.toSortedSet)
            exampleConfig.locusGrouping shouldEqual expLocusGrouping
        }
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
                case Left(messages) => messages.count(_ === "Missing proximityFilterStrategy section!") shouldEqual 1
            }
        }
    }

    test("Trivial region grouping with groups specified is an error.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        
        def mygen = for {
            (seq, optLocusGrouping, exclusions) <- genValidSeqAndLocusGroupOptAndExclusions(PositiveInt(10))
            (semantic, expectedMessage) <- Gen.oneOf(
                "UniversalProximityPermission" -> s"For universal proximity permission, both threshold and groups must be absent.", 
                "UniversalProximityProhibition" -> "For universal proximity prohibition, threshold must be present and groups must be absent.",
                )
            proximityFilterStrategy <- genValidParitionForRegionGrouping(seq)
        } yield (seq, optLocusGrouping, semantic, expectedMessage, proximityFilterStrategy, exclusions)
        
        forAll (mygen, arbitrary[NonnegativeReal]) { case ((seq, optLocusGrouping, semantic, expectedMessage, proximityFilterStrategy, exclusions), rawThreshold) => 
            val baseData: Map[String, ujson.Value] = {
                val records: NonEmptyList[ujson.Obj] = Random.shuffle(seq.allRounds.map(ImagingRound.roundToJsonObject).toList).toList.toNel.get
                Map(
                    "imagingRounds" -> ujson.Arr(records.toList*),
                    "proximityFilterStrategy" -> ujson.Obj(
                        "semantic" -> ujson.Str(semantic), 
                        "minimumPixelLikeSeparation" -> ujson.Num(rawThreshold),
                        "groups" -> proximityFilterStrategyToJson(proximityFilterStrategy.map(_.toList))
                    )
                )
            }
            val data: Map[String, ujson.Value] = addLocusGroupingAndExclusions(baseData, optLocusGrouping, exclusions)
            ImagingRoundsConfiguration.fromJsonMap(data) match {
                case Right(_) => fail(s"Expected parse failure on account presence of groups with trivial semantic, but it succeeded!")
                case Left(messages) => 
                    val numMatchMessages = messages.count(_ === expectedMessage).toInt
                    if numMatchMessages === 1 then succeed else {
                        println(s"MESSAGES: ${messages.mkString_("; ")}")
                        println(s"DATA (below)\n${write(data, indent = 4)}")
                        fail(s"Expected exactly 1 matching message but got $numMatchMessages. Data and messages are above.")
                    }
            }
        }
    }

    test("Region grouping missing semantic gives error, regardless of presence of groups or minimum separation distance.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        
        def mygen = for {
            (seq, optLocusGrouping, exclusions) <- genValidSeqAndLocusGroupOptAndExclusions(PositiveInt(10))
            optRegionGrouping <- Gen.option(genValidParitionForRegionGrouping(seq))
            optRawThreshold <- optRegionGrouping match {
                // Ensure that at least the region grouping or the raw threshold is nonempty.
                // That way, the region grouping section--already here we're omitting the semantic--will be nonempty.
                case None => arbitrary[NonnegativeReal].map(_.some)
                case _ => Gen.option(arbitrary[NonnegativeReal])
            }
        } yield (seq, optLocusGrouping, optRegionGrouping, optRawThreshold, exclusions)
        
        forAll (mygen) { (seq, optLocusGrouping, optRegionGrouping, optRawThreshold, exclusions) => 
            val baseData = {
                val records: NonEmptyList[ujson.Obj] = Random.shuffle(seq.allRounds.map(ImagingRound.roundToJsonObject).toList).toList.toNel.get
                val proximityFilterStrategyJsonData = List(
                    optRawThreshold.map(t => "minimumPixelLikeSeparation" -> ujson.Num(t)),
                    optRegionGrouping.map(g => "groups" -> proximityFilterStrategyToJson(g.map(_.toList)))
                ).flatten match {
                    case Nil => throw new IllegalStateException("Either optional threshold or optional grouping is empty!")
                    case kv1 :: rest => ujson.Obj(kv1, rest*)
                }
                Map("imagingRounds" -> ujson.Arr(records.toList*), "proximityFilterStrategy" -> proximityFilterStrategyJsonData)
            }
            val data: Map[String, ujson.Value] = addLocusGroupingAndExclusions(baseData, optLocusGrouping, exclusions)
            
            ImagingRoundsConfiguration.fromJsonMap(data) match {
                case Right(_) => fail(s"Expected parse failure on account presence of groups with trivial semantic, but it succeeded!")
                case Left(messages) => 
                    val expectMessage = "Missing semantic for proximity filter config section!"
                    val numMatchMessages = messages.count(_ === expectMessage).toInt
                    if numMatchMessages === 1 then succeed else {
                        println(s"MESSAGES: ${messages.mkString_("; ")}")
                        println(s"DATA (below)\n${write(data, indent = 4)}")
                        fail(s"Expected exactly 1 matching message ($expectMessage) but got $numMatchMessages. Data and messages are above.")
                    }
            }
        }
    }

    test("With or without other elements present, invalid semantic is an error.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        type BadSemanticValue = String // Give context-specific meaning to bare String.
        type RawThreshold = NonnegativeReal

        def mygen: Gen[(ImagingSequence, BadSemanticValue, Option[NonEmptyList[NonEmptySet[Timepoint]]], Option[RawThreshold], Option[NonEmptyList[LocusGroup]], Set[Timepoint])] = for {
            (seq, locusGroupingOpt, exclusions) <- genValidSeqAndLocusGroupOptAndExclusions(PositiveInt(10))
            semantic <- Gen.alphaNumStr.suchThat{
                s => ! Set(
                    "UniversalProximityPermission",
                    "UniversalProximityProhibition",
                    "SelectiveProximityPermission",
                    "SelectiveProximityProhibition",
                    )
                    .contains(s.toLowerCase) 
                }
            optRegionGrouping <- Gen.option(genValidParitionForRegionGrouping(seq))
            optRawThreshold <- Gen.option(arbitrary[NonnegativeReal])
        } yield (seq, semantic, optRegionGrouping, optRawThreshold, locusGroupingOpt, exclusions)

        forAll (mygen, minSuccessful(1000)) { case (seq, semantic, optRegionGrouping, optRawThreshold, locusGroupingOpt, exclusions) => 
            val baseData: Map[String, ujson.Value] = {
                val records: NonEmptyList[ujson.Obj] = 
                    Random.shuffle(seq.allRounds.map(ImagingRound.roundToJsonObject).toList).toList.toNel.get
                val proximityFilterStrategyJsonData = ujson.Obj(
                    "semantic" -> ujson.Str(semantic),
                    List(
                        optRawThreshold.map(t => "minimumPixelLikeSeparation" -> ujson.Num(t)), 
                        optRegionGrouping.map(g => "groups" -> proximityFilterStrategyToJson(g.map(_.toList))),
                        ).flatten*
                )
                Map("imagingRounds" -> ujson.Arr(records.toList*), "proximityFilterStrategy" -> proximityFilterStrategyJsonData)
            }
            val data: Map[String, ujson.Value] = addLocusGroupingAndExclusions(baseData, locusGroupingOpt, exclusions)
            
            ImagingRoundsConfiguration.fromJsonMap(data) match {
                case Right(_) => fail(s"Expected parse failure based on semantic '$semantic', but it succeeded!")
                case Left(messages) => 
                    val expMsg = s"Illegal value for proximity filter semantic: $semantic"
                    val numMatchMessages = messages.count(_ === expMsg).toInt
                    if numMatchMessages === 1 then succeed
                    else {
                        println(s"MESSAGES: ${messages.mkString_("; ")}")
                        println(s"DATA (below)\n${write(data, indent = 4)}")
                        fail(s"Expected exactly 1 matching message ($expMsg) but got $numMatchMessages. Data and messages are above.")
                    }
            }
        }
    }

    test("Without groups present, region grouping semantic must be trivial.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        type RawSemanticValue = String // Give context-specific meaning to bare String.
        type RawThreshold = PositiveReal

        def mygen: Gen[(ImagingSequence, RawSemanticValue, RawThreshold, Option[NonEmptyList[LocusGroup]], Set[Timepoint], String)] = for {
            (seq, locusGroupingOpt, exclusions) <- genValidSeqAndLocusGroupOptAndExclusions(PositiveInt(10))
            (semantic, expectedMessage) <- Gen.oneOf(
                Gen.oneOf(
                    "SelectiveProximityPermission" -> "selective proximity permission", 
                    "SelectiveProximityProhibition" -> "selective proximity prohibition",
                ).map{ (s, nameForErrText) => 
                    val expErrMsg = s"For $nameForErrText, threshold and grouping must be present."
                    (s, expErrMsg) 
                },
                Gen.alphaNumStr
                    .suchThat{ s =>  
                        val knowns = Set(
                            "UniversalProximityPermission",
                            "UniversalProximityProhibition",
                            "SelectiveProximityPermission", 
                            "SelectiveProximityProhibition",
                            )
                        !knowns.contains(s.toLowerCase)
                    }
                    .map(s => s -> s"Illegal value for proximity filter semantic: $s"),
            )
            threshold <- arbitrary[PositiveReal]
        } yield (seq, semantic, threshold, locusGroupingOpt, exclusions, expectedMessage)

        forAll (mygen, minSuccessful(1000)) { case (seq, semantic, rawThreshold, locusGroupingOpt, exclusions, expectedMessage) => 
            val baseData: Map[String, ujson.Value] = {
                val records: NonEmptyList[ujson.Obj] = 
                    Random.shuffle(seq.allRounds.map(ImagingRound.roundToJsonObject).toList).toList.toNel.get
                val proximityFilterStrategyJsonData = ujson.Obj("semantic" -> ujson.Str(semantic), "minimumPixelLikeSeparation" -> ujson.Num(rawThreshold))
                Map("imagingRounds" -> ujson.Arr(records.toList*), "proximityFilterStrategy" -> proximityFilterStrategyJsonData)
            }
            val data: Map[String, ujson.Value] = addLocusGroupingAndExclusions(baseData, locusGroupingOpt, exclusions)
            
            ImagingRoundsConfiguration.fromJsonMap(data) match {
                case Right(_) => fail(s"Expected parse failure based on semantic '$semantic', but it succeeded!")
                case Left(messages) => 
                    val numMatchMessages = messages.count(_ === expectedMessage).toInt
                    if numMatchMessages === 1 then succeed
                    else
                        println(s"MESSAGES: ${messages.mkString_("; ")}")
                        println(s"DATA (below)\n${write(data, indent = 4)}")
                        fail(s"Expected exactly 1 matching message ($expectedMessage) but got $numMatchMessages. Data and messages are above.")
            }
        }
    }

    test("Nontrivial region grouping must specify groups that constitute a partition of regional round timepoints from the imaging sequence.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        val maxTime = 100
        given arbTime: Arbitrary[Timepoint] = 
            // Limit range to encourage overlap b/w regional rounds in sequence and grouping.
            Gen.choose(0, maxTime).map(Timepoint.unsafe).toArbitrary

        def mygen = for {
            (seq, optLocusGrouping, exclusions) <- genValidSeqAndLocusGroupOptAndExclusions(PositiveInt.unsafe(maxTime / 2))
            regionalTimes = seq.regionRounds.map(_.time).toList.toSet
            semantic <- Gen.oneOf("SelectiveProximityPermission", "SelectiveProximityProhibition")
            (regionGroups, findExpMessages) <- Gen.nonEmptyListOf{ Gen.nonEmptyListOf(arbitrary[Timepoint]).map(_.toNel.get) }
                .map(_.toNel.get)
                // Ensure that subsets fail uniqueness, disjointness, fail to cover regionals, or fail to be covered by regionals
                // In the process, determine the expected error message.
                .map{ gs => (for {
                    // First, try for duplicates within an alleged subset of the grouping.
                    _ <- gs.toList.traverse{ ts => 
                        if ts.toList.toSet.size === ts.length then ().asRight
                        else NonEmptyList.one{(
                            "Not proper sets",
                            (_: String).endsWith("repeated items for regional group")
                        )}.asLeft
                    }
                    _ <- {
                        // No duplicates within subsets, so try to find overlap between them.
                        // In the process, determine the flattened set of timepoints in the grouping.
                        val (isDisjoint, seen) = gs.toList.foldLeft(false -> Set.empty[Timepoint]){ case ((isDisjoint, acc), g) => 
                            val ts = g.toList.toSet
                            (isDisjoint || (ts & acc).nonEmpty, acc ++ ts)
                        }
                        // Determine which error messages to search for.
                        List(
                            isDisjoint.option{(
                                "Overlapping subsets",
                                (_: String) === "Proximity filter strategy's subsets are not disjoint!"
                            )},
                            (seen -- regionalTimes).nonEmpty.option{(
                                "Not in sequence",
                                (_: String).startsWith(s"Unknown timepoint(s) (proximity filter's grouping (rel. to regionals in imaging sequence))")
                            )},
                            (regionalTimes -- seen).nonEmpty.option{(
                                "Not in grouping", 
                                (_: String).startsWith(s"Unknown timepoint(s) (regionals in imaging sequence (rel. to proximity filter strategy))")
                            )}
                        ).flatten.toNel.toLeft(())
                    }
                } yield ()).leftMap(gs -> _) }
                .suchThat(_.isLeft)
                .map(_.swap.getOrElse{ throw new IllegalStateException("Allegedly generated a Left but then failed to get value after .swap!") })
        } yield (seq, optLocusGrouping, semantic, regionGroups, exclusions, findExpMessages)
        
        forAll (mygen, arbitrary[NonnegativeReal], minSuccessful(10000)) {
            case ((seq, optLocusGrouping, semantic, regionGroups, exclusions, findExpMessages), threshold) =>
                val baseData: Map[String, ujson.Value] = {
                    val records: NonEmptyList[ujson.Obj] = 
                        Random.shuffle(seq.allRounds.map(ImagingRound.roundToJsonObject).toList).toList.toNel.get
                    val proximityFilterStrategyJsonData = ujson.Obj(
                        "semantic" -> ujson.Str(semantic), 
                        "groups" -> proximityFilterStrategyToJson(regionGroups.map(_.toList)),
                        "minimumPixelLikeSeparation" -> ujson.Num(threshold)
                    )
                    Map("imagingRounds" -> ujson.Arr(records.toList*), "proximityFilterStrategy" -> proximityFilterStrategyJsonData)
                }
                val data: Map[String, ujson.Value] = addLocusGroupingAndExclusions(baseData, optLocusGrouping, exclusions)

                ImagingRoundsConfiguration.fromJsonMap(data) match {
                    case Right(_) => 
                        println(s"GROUPING: $regionGroups")
                        println(s"REGIONAL TIMES: ${seq.regionRounds.map(_.time).mkString_(", ")}")
                        fail("Expected parse failure for non-partition of imaging sequence's regional rounds, but got success. Data are above.")
                    case Left(messages) => 
                        val unmetChecks = findExpMessages.filterNot{ (testName, check) => messages.count(check) === 1 }.map(_._1)
                        if unmetChecks.isEmpty then succeed else {
                            println(s"MESSAGES: ${messages.mkString_("; ")}")
                            println(s"DATA (below)\n${write(data, indent = 4)}")
                            fail(s"Unmet check(s): ${unmetChecks.mkString("; ")}")
                        }
                }
        }
    }

    test("Locus grouping must either be absent or be identical, in union over timepoints, to timepoints from locus imaging rounds from imaging sequence.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        val maxTime = 100
        given arbTime: Arbitrary[Timepoint] = 
            // Limit range to encourage overlap b/w regional rounds in sequence and grouping.
            Gen.choose(0, maxTime).map(Timepoint.unsafe).toArbitrary

        def mygen = for {
            // First, generate the basic imaging rounds sequence, with (maybe empty) locus grouping, and round exclusions.
            (seq, optLocusGrouping, exclusions) <- genValidSeqAndLocusGroupOptAndExclusions(PositiveInt.unsafe(maxTime / 2))
            /* Collect all regional and local timepoint for set-theoretic comparisons. */
            regionalTimes = seq.regionRounds.map(_.time).toList.toSet
            locusTimes = seq.locusRounds.map(_.time).toSet
            /* Determine the final locus grouping, and method call expectation (Right-wrapped () for success, Left-wrapped NEL of tests o/w). */
            (modLocusGrouping, altMessagesTests) <- {
                optLocusGrouping match {
                    case None => Gen.oneOf(
                        Gen.const(None -> ().asRight), // Absence of locus grouping is always valid.
                        {
                            // Here, we generate dummy "locus" timepoints and assign to one of the real regional times.
                            val gen = for {
                                ts <- Gen.nonEmptyListOf(arbitrary[Timepoint].suchThat(t => !locusTimes.contains(t)))
                                r <- Gen.oneOf(regionalTimes)
                            } yield NonEmptyList.one(LocusGroup(r, ts.toNel.get.toNes))
                            gen.map{ g => 
                                // Since each timepoint is generated to NOT be in the imaging sequence, anything added here is to be expected as extra.
                                val numExtra = g.head.locusTimepoints.size
                                val f = { (_: String).startsWith(s"$numExtra timepoint(s) in locus grouping and not found as locus imaging timepoints") }
                                g.some -> NonEmptyList.one("Extra in grouping 1" -> f).asLeft[Unit]
                            }
                        }
                    )
                    case Some(groups) => Gen.oneOf(
                        Gen.const{ (None, ().asRight) }, // Absence of locus grouping is always valid.
                        if groups.length === 1 && groups.head.locusTimepoints.size === 1 
                        then Gen.const{
                            // There's only 1 group, and it has 1 value...
                            val msgTest = 
                                // ...If it's a real locus timepoint, then the grouping is valid...
                                if locusTimes.contains(groups.head.locusTimepoints.toNonEmptyList.head)
                                then ().asRight
                                // ...otherwise, we have an invalid grouping.
                                else {
                                    val prefix = "1 locus timepoint(s) in imaging sequence and not found in locus grouping"
                                    NonEmptyList.one{ ("Uncovered in sequence 1", (_: String).startsWith(prefix)) }.asLeft[Unit]
                                }
                            groups.some -> msgTest
                        }
                        else for {
                            // We have at least 1 group and more than 1 locus timepoint in total, so we can 
                            // definitely generate at least 1 to remove and still have a structurally (nonempty-of-nonempty) grouping.
                            // First, remove a single locus timepoint in the imaging rounds sequence from the locus grouping.
                            subtracted <- Gen.oneOf(locusTimes)
                                .map{ t => groups.toList.foldRight(List.empty[LocusGroup]){
                                    case (g, acc) => (g.locusTimepoints - t).toNes.fold(acc)(ts => g.copy(locusTimepoints = ts) :: acc)
                                } }
                            // Then, add a new timepoint to each group.
                            // NB: here we seemingly don't check for possibility of introducing non-disjointness among the subgroups.
                            subtractedAndAdded <- subtracted.toList.traverse{ g => 
                                arbitrary[Timepoint]
                                    .suchThat(t => !locusTimes.contains(t))
                                    .map(t => g.copy(locusTimepoints = g.locusTimepoints.add(t)))
                            }
                            // Since we've both removed and added timepoints to the grouping that originally partitioned 
                            // the locus timepoints from the imaging rounds sequence, we expect both kinds of error message.
                            missingPointTests = NonEmptyList.of(
                                ("Uncovered in sequence 2", (_: String).startsWith("1 locus timepoint(s) in imaging sequence and not found in locus grouping")), 
                                ("Extra in grouping 3", (_: String).startsWith(s"${subtractedAndAdded.length} timepoint(s) in locus grouping and not found as locus imaging timepoints"))
                            )
                        } yield (subtractedAndAdded.toNel.get.some, missingPointTests.asLeft[Unit])
                    )
                }
            }
            semantic <- Gen.oneOf(
                "UniversalProximityPermission",
                "UniversalProximityProhibition",
                "SelectiveProximityPermission",
                "SelectiveProximityProhibition",
                )
            regionGroups <- genValidParitionForRegionGrouping(seq)
        } yield (seq, modLocusGrouping, semantic, regionGroups, exclusions, altMessagesTests)
        
        forAll (mygen, arbitrary[NonnegativeReal], minSuccessful(10000)) {
            case ((seq, optLocusGrouping, semantic, regionGroups, exclusions, altMessagesTests), threshold) =>
                val baseData: Map[String, ujson.Value] = {
                    val records: NonEmptyList[ujson.Obj] = 
                        Random.shuffle(seq.allRounds.map(ImagingRound.roundToJsonObject).toList).toList.toNel.get
                    val extra = semantic match {
                        case ("UniversalProximityPermission" | "UniversalProximityProhibition") => List()
                        case ("SelectiveProximityPermission" | "SelectiveProximityProhibition") => 
                            List("groups" -> proximityFilterStrategyToJson(regionGroups.map(_.toList)))
                    }
                    val thresholdKV = 
                        if semantic === "UniversalProximityPermission" then List.empty
                        else List("minimumPixelLikeSeparation" -> ujson.Num(threshold))
                    val proximityFilterStrategyJsonData = ujson.Obj(
                        "semantic" -> ujson.Str(semantic), 
                        (thresholdKV ++ extra)*,
                    )
                    Map("imagingRounds" -> ujson.Arr(records.toList*), "proximityFilterStrategy" -> proximityFilterStrategyJsonData)
                }
                val data: Map[String, ujson.Value] = addLocusGroupingAndExclusions(baseData, optLocusGrouping, exclusions)

                (ImagingRoundsConfiguration.fromJsonMap(data), altMessagesTests) match {
                    case (Right(_), Right(_)) => succeed
                    case (Right(_), Left(tests)) => 
                        println(s"DATA (below)\n${write(data, indent = 4)}")
                        fail(s"Got parse success when expecting failure(s) for these reason(s): ${tests.map(_._1).mkString_("; ")}")
                    case (Left(messages), Right(_)) => 
                        println(s"MESSAGES: ${messages.mkString_("; ")}")
                        println(s"DATA (below)\n${write(data, indent = 4)}")
                        fail(s"Expected parse success but got failure(s): ${messages.mkString_("; ")}")
                    case (Left(messages), Left(tests)) => 
                        val unmetChecks = tests.filterNot{ (testName, check) => messages.count(check) === 1 }.map(_._1)
                        if unmetChecks.isEmpty then succeed else {
                            println(s"MESSAGES: ${messages.mkString_("; ")}")
                            println(s"DATA (below)\n${write(data, indent = 4)}")
                            fail(s"Unmet check(s): ${unmetChecks.mkString("; ")}")
                        }
                }
        }
    }

    test("Each of the locus grouping's keys must be a regional round timepoint from the imaging sequence") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        val maxTime = 100
        given arbTime: Arbitrary[Timepoint] = 
            // Limit range to encourage overlap b/w regional rounds in sequence and grouping.
            Gen.choose(0, maxTime).map(Timepoint.unsafe).toArbitrary

        def mygen = for {
            (seq, locusGrouping, exclusions) <- genValidSeqAndLocusGroupOptAndExclusions(PositiveInt.unsafe(maxTime / 2))
                .suchThat(_._2.nonEmpty)
                .map((seq, optGrouping, exclusions) => (seq, optGrouping.get, exclusions))
            badRegionKey <- arbitrary[Timepoint].suchThat{ t => !seq.regionRounds.map(_.time).toList.toSet.contains(t) }
            semantic <- Gen.oneOf(
                "UniversalProximityPermission",
                "UniversalProximityProhibition",
                "SelectiveProximityPermission",
                "SelectiveProximityProhibition",
                )
            regionGroups <- genValidParitionForRegionGrouping(seq)
        } yield (seq, locusGrouping, exclusions, semantic, regionGroups, badRegionKey)

        forAll (mygen, arbitrary[NonnegativeReal]) { 
            case ((seq, locusGrouping, exclusions, semantic, regionGroups, badRegionKey), threshold) =>
                val baseData: Map[String, ujson.Value] = {
                    val records: NonEmptyList[ujson.Obj] = 
                        Random.shuffle(seq.allRounds.map(ImagingRound.roundToJsonObject).toList).toList.toNel.get
                    val extra = semantic match {
                        case ("UniversalProximityPermission" | "UniversalProximityProhibition") => List()
                        case ("SelectiveProximityPermission" | "SelectiveProximityProhibition") => 
                            List("groups" -> proximityFilterStrategyToJson(regionGroups.map(_.toList)))
                    }
                    val thresholdKV = 
                        if semantic === "UniversalProximityPermission" then List.empty
                        else List("minimumPixelLikeSeparation" -> ujson.Num(threshold))
                    val proximityFilterStrategyJsonData = ujson.Obj(
                        "semantic" -> ujson.Str(semantic), 
                        (thresholdKV ++ extra)*,
                    )
                    Map("imagingRounds" -> ujson.Arr(records.toList*), "proximityFilterStrategy" -> proximityFilterStrategyJsonData)
                }
                val data: Map[String, ujson.Value] = {
                    val newLocusGrouping = NonEmptyList(
                        locusGrouping.head.copy(regionalTimepoint = badRegionKey), 
                        locusGrouping.tail
                    )
                    addLocusGroupingAndExclusions(baseData, newLocusGrouping.some, exclusions)
                }

                ImagingRoundsConfiguration.fromJsonMap(data) match {
                    case Right(_) => 
                        println(s"DATA (below)\n${write(data, indent = 4)}")
                        fail("Parse succeeded when failure was expected!")
                    case Left(messages) => 
                        val expectMessage = "1 timepoint(s) as keys in locus grouping that aren't regional."
                        val numMatchMessages = messages.count(_ === expectMessage).toInt
                        if numMatchMessages === 1 then succeed
                        else {
                            println(s"MESSAGES: ${messages.mkString_("; ")}")
                            println(s"DATA (below)\n${write(data, indent = 4)}")
                            fail(s"Expected exactly 1 matching message ($expectMessage) but got $numMatchMessages. Data and messages are above.")
                        }
                }
        }
    }

    test("Any timepoint to exclude from tracing must be a timepoint in the imaging sequence.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        val maxTime = 100
        given arbTime: Arbitrary[Timepoint] = 
            // Limit range to encourage overlap b/w regional rounds in sequence and grouping.
            Gen.choose(0, maxTime).map(Timepoint.unsafe).toArbitrary

        def mygen = for {
            (seq, optLocusGrouping, _) <- genValidSeqAndLocusGroupOptAndExclusions(PositiveInt.unsafe(maxTime / 2))
            semantic <- Gen.oneOf(
                "UniversalProximityPermission",
                "UniversalProximityProhibition",
                "SelectiveProximityPermission",
                "SelectiveProximityProhibition",
                )
            regionGroups <- genValidParitionForRegionGrouping(seq)
            exclusions <- arbitrary[List[Timepoint]].suchThat(_.exists(t => !seq.allTimepoints.contains(t)))
        } yield (seq, optLocusGrouping, semantic, regionGroups, exclusions.toSet)

        forAll (mygen, arbitrary[NonnegativeReal]) { 
            case ((seq, optLocusGrouping, semantic, regionGroups, exclusions), threshold) => 
                val baseData: Map[String, ujson.Value] = {
                    val records: NonEmptyList[ujson.Obj] = 
                        Random.shuffle(seq.allRounds.map(ImagingRound.roundToJsonObject).toList).toList.toNel.get
                    val extra = semantic match {
                        case ("UniversalProximityPermission" | "UniversalProximityProhibition") => List()
                        case ("SelectiveProximityPermission" | "SelectiveProximityProhibition") => 
                            List("groups" -> proximityFilterStrategyToJson(regionGroups.map(_.toList)))
                    }
                    val thresholdKV = 
                        if semantic === "UniversalProximityPermission" then List.empty
                        else List("minimumPixelLikeSeparation" -> ujson.Num(threshold))
                    val proximityFilterStrategyJsonData = ujson.Obj(
                        "semantic" -> ujson.Str(semantic), 
                        (thresholdKV ++ extra)*,
                    )
                    Map("imagingRounds" -> ujson.Arr(records.toList*), "proximityFilterStrategy" -> proximityFilterStrategyJsonData)
                }
                val data: Map[String, ujson.Value] = addLocusGroupingAndExclusions(baseData, optLocusGrouping, exclusions)

                ImagingRoundsConfiguration.fromJsonMap(data) match {
                    case Right(_) => 
                        println(s"DATA (below)\n${write(data, indent = 4)}")
                        fail("Parse succeeded when failure was expected!")
                    case Left(messages) => 
                        val unknown = exclusions -- seq.allTimepoints.toList
                        val expectMessage = s"Unknown timepoint(s) (tracing exclusions): ${unknown.toList.sorted.map(_.show).mkString(", ")}"
                        val numMatchMessages = messages.count(_ === expectMessage).toInt
                        if numMatchMessages === 1 then succeed
                        else {
                            println(s"MESSAGES: ${messages.mkString_("; ")}")
                            println(s"DATA (below)\n${write(data, indent = 4)}")
                            fail(s"Expected exactly 1 matching message ($expectMessage) but got $numMatchMessages. Data and messages are above.")
                        }
                }
        }
    }
    
    given rwForSeq: ReadWriter[ImagingSequence] = 
            ImagingRoundsConfiguration.rwForImagingSequence(using ImagingRound.rwForImagingRound)
    
    given rwForTime: ReadWriter[Timepoint] = readwriter[ujson.Value].bimap(time => ujson.Num(time.get), json => Timepoint.unsafe(json.int))

    private def proximityFilterStrategyToJson(grouping: NonEmptyList[List[Timepoint]]): ujson.Value = 
        ujson.Arr(grouping.toList.map(ts => ujson.Arr(ts.map(t => ujson.Num(t.get))*))*)

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
        seq <- genTimeValidSequence(Gen.choose(1, maxNumRounds).map(PositiveInt.unsafe))
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

    private def genValidParitionForLocusGrouping(sequence: ImagingSequence): Gen[Option[NonEmptyList[NonEmptySet[Timepoint]]]] = {
        val loci = sequence.locusRounds
        if loci.isEmpty then Gen.const(None)
        else Gen.option{ 
            Gen.choose(1, loci.length)
                .flatMap{ k => Random.shuffle(loci).toList.splitAt(k) }
                .map{ (g1, g2) => List(g1, g2).flatMap(_.toNel).map(_.map(_.time).toNes).toNel.get }
        }
    }

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
end TestImagingRoundsConfigurationUnderAssumptionOfDisjointnessofGeneratedLocusGrouping
