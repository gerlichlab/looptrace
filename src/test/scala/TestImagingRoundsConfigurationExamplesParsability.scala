package at.ac.oeaw.imba.gerlich.looptrace

import scala.collection.SortedSet
import scala.util.{ Failure, Success, Try }
import scala.util.chaining.*
import cats.Eq
import cats.data.{ NonEmptyList, NonEmptySet }
import cats.syntax.all.*
import mouse.boolean.*
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.{
    BuildError,
    LocusGroup, 
    RoiPartnersRequirementType,
    SelectiveProximityPermission, 
    SelectiveProximityProhibition, 
    TraceIdDefinitionRule,
    UniversalProximityPermission, 
    UniversalProximityProhibition, 
}
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanDistance
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.ProximityGroup
import cats.data.Validated.Invalid
import cats.data.Validated.Valid

/** Tests of examples of imaging rounds config files */
class TestImagingRoundsConfigurationExamplesParsability extends AnyFunSuite with ScalaCheckPropertyChecks with should.Matchers:

    test("proximityFilterStrategy is correct for strategies other than SelectiveProximityPermission") {
        val expMinSep = NonnegativeReal(5.0)
        val expGrouping = NonEmptyList.of(NonEmptySet.of(8, 9), NonEmptySet.of(10, 11)).map(_.map(ImagingTimepoint.unsafe))
        forAll (Table(
            ("subfolder", "configFileName", "expectation"),
            ("UniversalProximityPermission", "example__imaging_rounds_configuration__universal_permission.json", UniversalProximityPermission),
            ("UniversalProximityProhibition", "example__imaging_rounds_configuration__universal_prohibition.json", UniversalProximityProhibition(expMinSep)),
            ("SelectiveProximityProhibition", "example__imaging_rounds_configuration__selective_prohibition.json", SelectiveProximityProhibition(expMinSep, expGrouping)),
        )) { (subfolder, filename, expectation) =>
            val exampleConfig: ImagingRoundsConfiguration = {
                val configFile = getResourcePath(subfolder = subfolder, filename = filename)
                ImagingRoundsConfiguration.unsafeFromJsonFile(configFile)
            }
            exampleConfig.proximityFilterStrategy shouldEqual expectation
        }
    }

    test("SelectiveProximityPermission example imaging rounds config files parse correctly.") {
        val expectedNonemptyLocusGrouping = 
            NonEmptySet.of(
                8 -> NonEmptySet.of(1, 6),
                9 -> NonEmptySet.one(2), 
                10 -> NonEmptySet.of(3, 4),
                11 -> NonEmptySet.one(5)
            )
            .map{ (regTime, locusTimes) => LocusGroup(ImagingTimepoint.unsafe(regTime), locusTimes.map(ImagingTimepoint.unsafe)) }
        forAll (Table(
            ("subfolder", "configFileName", "maybeExpectedLocusGrouping"), 
            ("SelectiveProximityPermission", "example_imaging_rounds_configuration.json", expectedNonemptyLocusGrouping.some),
            ("SelectiveProximityPermission", "rounds_config_with_empty_locus_grouping.json", None), 
            ("SelectiveProximityPermission", "rounds_config_with_null_locus_grouping.json", None), 
            ("SelectiveProximityPermission", "rounds_config_without_locus_grouping.json", None), 
        )) { (subfolder, configFileName, maybeExpectedLocusGrouping) => 
            val exampleConfig: ImagingRoundsConfiguration = {
                val configFile = getResourcePath(subfolder = subfolder, filename = configFileName)
                ImagingRoundsConfiguration.unsafeFromJsonFile(configFile)
            }
            exampleConfig.numberOfRounds shouldEqual 12
            exampleConfig.proximityFilterStrategy shouldEqual SelectiveProximityPermission(
                NonnegativeReal(5.0), 
                NonEmptyList.of(NonEmptySet.of(8, 9), NonEmptySet.of(10, 11)).map(_.map(ImagingTimepoint.unsafe))
            )
            exampleConfig.tracingExclusions shouldEqual Set(0, 8, 9, 10, 11).map(ImagingTimepoint.unsafe)
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

    test("Locus grouping validation can be strict or lax. Issue #295") {
        val countExpectedErrors = (_: NonEmptyList[String]).count(_.contains("locus timepoint(s) in imagingRounds and not found in locusGrouping (nor in tracingExclusions)"))
        
        forAll (Table(
            ("subfolder", "filename", "expectError"), 
            // This first example omits some locus imaging timepoints from the values of the locusGrouping section, 
            // but it sets the flag for validation of locus timepoints covering (checkLocusTimepointCovering) to false.
            // Critically, some omitted locus timepoints are NOT in the tracingExclusions, as that's a separate (automatic) 
            // exception for a locus timepoint from the requirement to be present in at least one values list in the locusGrouping.
            ("LocusGroupingValidation", "example__imaging_rounds_configuration__selective_prohibition__no_check_locus_grouping_295.json", false), 
            /* These subsequent examples have the same inclusion/exclusion properties w.r.t. locus imaging timepoints and 
               the locusGrouping section, but the have the key for validation either missing or set to null or true, 
               so the validation should be done as declared (or as default). */
            ("LocusGroupingValidation", "fail__imaging_rounds_config__missing_locus_validation_295.json", true),
            ("LocusGroupingValidation", "fail__imaging_rounds_config__null_locus_validation_295.json", true),
            ("LocusGroupingValidation", "fail__imaging_rounds_config__true_locus_validation_295.json", true),
        )) { (subfolder, filename, expectError) => 
            val configFile = getResourcePath(subfolder = subfolder, filename = filename)
            val safeParseResult = ImagingRoundsConfiguration.fromJsonFile(configFile)
            if expectError then {
                safeParseResult match {
                    case Left(errorMessages) => countExpectedErrors(errorMessages) shouldEqual 1
                    case Right(_) => fail(s"Expected parse failure for $configFile but succeeded.")
                }
                val error = intercept[BuildError.FromJsonFile]{ ImagingRoundsConfiguration.unsafeFromJsonFile(configFile) }
                countExpectedErrors(error.messages) shouldEqual 1
            } else {
                safeParseResult.isRight shouldBe true
                Try{ ImagingRoundsConfiguration.unsafeFromJsonFile(configFile)} match {
                    case Success(_) => succeed
                    case Failure(exception) => fail(s"Expected parse success for $configFile but failed: $exception")
                }
            }
        }
    }

    test("locusGrouping tolerates absence of locus imaging timepoints which are in the tracingExclusions. Issue #304") {
        forAll (Table(
            ("subfolder", "filename", "expectError"), 
            /* Critically, these examples DO NOT SET THE VALIDATION flag, so validation is done as the default behavior. 
               This means that it's the behavior of the validation w.r.t. the tracingExclusions' relation to the relation 
               between the locusGrouping and the locus timepoints and the imagingRounds which is under test.
               So we expect failure when the locus timepoints omitted from the locusGrouping are not in the tracingExclusions, 
               but we expect success when these omitted timepoints are exempted from validation by virtue of their inclusion 
               in the tracingExclusions listing. */
            ("LocusGroupingValidation", "example__rounds_config__with_times_correctly_omitted_from_locus_grouping__304.json", false), 
            ("LocusGroupingValidation", "fail__rounds_config__with_times_incorrectly_omitted_from_locus_grouping__304.json", true), 
        )) { (subfolder, filename, expectError) =>
            val configFile = getResourcePath(subfolder = subfolder, filename = filename)
            val safeParseResult = ImagingRoundsConfiguration.fromJsonFile(configFile)
            if expectError then {
                safeParseResult match {
                    case Left(errorMessages) => 
                        errorMessages.count(_ === "2 locus timepoint(s) in imagingRounds and not found in locusGrouping (nor in tracingExclusions): 3, 4") shouldEqual 1
                    case Right(_) => fail(s"Expected parse failure for $configFile but succeeded.")
                }
                
            } else {
                safeParseResult.isRight shouldBe true
                Try{ ImagingRoundsConfiguration.unsafeFromJsonFile(configFile)} match {
                    case Success(_) => succeed
                    case Failure(exception) => fail(s"Expected parse success for $configFile but failed: $exception")
                }
            }
        }
    }

    test("proximityFilterStrategy grouping cannot have any regional timepoint in any values list for groups.") {
        forAll (Table(
            ("subfolder", "filename", "expectedExtra"), 
            (
                "LocusGroupingValidation", 
                "fail__rounds_config_with_different_regional_timepoint_in_locus_grouping_values__permission.json", 
                NonEmptySet.one("2 timepoint(s) in locus grouping and not found as locus imaging timepoints: 8, 10"),
            ), 
            (
                "LocusGroupingValidation", 
                "fail__rounds_config_with_different_regional_timepoint_in_locus_grouping_values__prohibition.json", 
                NonEmptySet.one("2 timepoint(s) in locus grouping and not found as locus imaging timepoints: 8, 10"),
            ), 
            (
                "LocusGroupingValidation", 
                "fail__rounds_config_with_same_regional_timepoint_in_locus_grouping_values__permission.json", 
                NonEmptySet.of(
                    "Regional time 9 is contained in its own locus times group!",
                    "Regional time 11 is contained in its own locus times group!",
                ),
            ), 
            (
                "LocusGroupingValidation", 
                "fail__rounds_config_with_same_regional_timepoint_in_locus_grouping_values__prohibition.json", 
                NonEmptySet.of(
                    s"Regional time 9 is contained in its own locus times group!",
                    s"Regional time 11 is contained in its own locus times group!",
                ),
            ), 
            (
                "LocusGroupingValidation", 
                "fail__rounds_config_with_unmapped_regional_timepoint_in_locus_grouping_values__permission.json", 
                NonEmptySet.one("1 timepoint(s) in locus grouping and not found as locus imaging timepoints: 9"),
            ), 
            (
                "LocusGroupingValidation", 
                "fail__rounds_config_with_unmapped_regional_timepoint_in_locus_grouping_values__prohibition.json", 
                NonEmptySet.one("1 timepoint(s) in locus grouping and not found as locus imaging timepoints: 9"),
            ), 
        )) { (subfolder, filename, expectedMessages) =>
            val configFile = getResourcePath(subfolder = subfolder, filename = filename)
            val safeParseResult = ImagingRoundsConfiguration.fromJsonFile(configFile)
            safeParseResult match {
                case Left(errorMessages) => expectedMessages.filterNot(errorMessages.contains_).isEmpty shouldBe true
                case Right(_) => fail(s"Expected parse failure for $configFile but succeeded.")
            }
        }
    }

    test("Simple canonical example of merge groups for tracing parses as expected."):
        val configFile = getResourcePath(
            subfolder = "DifferentTimepointRegionalMergeForTracing", 
            filename = "good_example__legitimate_tracing_merge_groups.json",
        )
        val expectedSuccessCases = 
            val expRules = 
                val expDistance = 
                    import io.github.iltotore.iron.autoRefine
                    EuclideanDistance.Threshold(NonnegativeReal(2000))
                NonEmptyList.of(
                    TraceGroupId.unsafe("A") -> MergeBag.unsafe(Set(6, 7)), 
                    TraceGroupId.unsafe("B") -> MergeBag.unsafe(Set(8, 9)),
                ).map{ (groupId, g) => 
                    TraceIdDefinitionRule(
                        groupId,
                        ProximityGroup(expDistance, g), 
                        RoiPartnersRequirementType.Conjunctive,
                    ) 
                }
            Table(
                ("configFileName", "expectedMergeRules"), 
                ("good_example__legitimate_tracing_merge_groups.json", expRules), 
                ("good_example__locus_time_overlap_prohibition_for_trace_merge_is_only_within_group.json", expRules),
            )
        val expectedFilterSetting = true
        forAll (expectedSuccessCases) { (configFileName, expectedMergeRules) => 
            val configFile = getResourcePath(
                subfolder = "DifferentTimepointRegionalMergeForTracing", 
                filename = configFileName,
            )
            checkParseSuccess(configFile, expectedFilterSetting -> expectedMergeRules)
        }

    test("Any non-regional timepoint listed for tracing merger is an error."):
        val expectedFailureCases = Table(
            ("configFileName", "expectedMessage"), 
            ("bad_example__locus_time_in_tracing_merge_groups.json", "1 non-regional time(s) in merge rules: 5"),
            ("bad_example__blank_time_in_tracing_merge_groups.json", "1 non-regional time(s) in merge rules: 0"),
        )
        forAll (expectedFailureCases) { (configFileName, expectedMessage) => 
            val configFile = getResourcePath(
                subfolder = "DifferentTimepointRegionalMergeForTracing", 
                filename = configFileName,
            )
            checkParseFailure(configFile, expectedMessage) // Test exact equality of message and expectation.
        }

    test("Any overlap of locus timepoint sets for regional timepoints to merge is an error. #384"):
        val configFile = getResourcePath(
            subfolder = "DifferentTimepointRegionalMergeForTracing", 
            filename = "bad_example__overlap_locus_times_for_regional_times_to_merge__384.json",
        )
        checkParseFailure(
            configFile, 
            "Regionals timepoints to merge for tracing map to overlapping locus timepoint sets",
            check = (msg: String, exp: String) => msg.startsWith(exp) // Here we just to prefix check.
        )

    private def checkParseSuccess(configFile: os.Path, expectedMergeParseResult: (Boolean, NonEmptyList[TraceIdDefinitionRule])) = 
        val (expFilter, expRules) = expectedMergeParseResult
        ImagingRoundsConfiguration.fromJsonFile(configFile) match {
            case Left(messages) => 
                fail(s"${messages.length} error message(s) parsing config file $configFile: ${messages.mkString_("; ")}")
            case Right(conf) => 
                val discardUngroupedMembersNel = 
                    val obsFilter = conf.discardRoisNotInGroupsOfInterest
                    if obsFilter === expFilter
                    then ().validNel
                    else s"Expected $expFilter for filter status but got $obsFilter ".invalidNel
                val mergeRulesNel = conf.mergeRules match {
                    case None => "No merge rules section".invalidNel
                    case Some(obsRules) => 
                        given Eq[TraceIdDefinitionRule] = Eq.fromUniversalEquals
                        if obsRules === expRules
                        then ().validNel
                        else f"Observed rules ($obsRules) don't match expected ($expRules)".invalidNel
                }
                (discardUngroupedMembersNel, mergeRulesNel).tupled match {
                    case Valid(a) => succeed
                    case Invalid(messages) => fail(f"${messages.length} failure(s): ${messages.mkString_("; ")}")
                }
        }

    object MergeBag:
        def unsafe = (_: Set[Int]).map(ImagingTimepoint.unsafe).pipe(AtLeast2.unsafe)

    private def checkParseFailure(configFile: os.Path, expectedMessage: String, check: (String, String) => Boolean = cats.Eq[String].eqv) = 
        ImagingRoundsConfiguration.fromJsonFile(configFile) match {
            case Left(messages) => 
                if messages.count(check(_, expectedMessage)) === 1
                then succeed
                else fail(s"No message parse fail message matched query ($expectedMessage); messages: ${messages.mkString_("; ")}")
            case Right(_) => 
                fail(s"Expected config parse to fail, but it succeeded on file $configFile")
        }

    private def getResourcePath(subfolder: String, filename: String): os.Path = 
        os.Path(getClass.getResource(s"/TestImagingRoundsConfiguration/$subfolder/$filename").getPath)

end TestImagingRoundsConfigurationExamplesParsability
