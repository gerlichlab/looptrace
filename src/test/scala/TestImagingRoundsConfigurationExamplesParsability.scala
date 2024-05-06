package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ Failure, Success, Try }
import cats.data.{ NonEmptyList, NonEmptySet }
import cats.syntax.all.*
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.{
    BuildError,
    LocusGroup, 
    SelectiveProximityPermission, 
    SelectiveProximityProhibition, 
    UniversalProximityPermission, 
    UniversalProximityProhibition, 
}

/** Tests of examples of imaging rounds config files */
class TestImagingRoundsConfigurationExamplesParsability extends AnyFunSuite with ScalaCheckPropertyChecks with should.Matchers:

    test("proximityFilterStrategy is correct for strategies other than SelectiveProximityPermission") {
        val expMinSep = PositiveReal(5.0)
        val expGrouping = NonEmptyList.of(NonEmptySet.of(8, 9), NonEmptySet.of(10, 11)).map(_.map(Timepoint.unsafe))
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
            .map{ (regTime, locusTimes) => LocusGroup(Timepoint.unsafe(regTime), locusTimes.map(Timepoint.unsafe)) }
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

    test("Locus grouping validation can be strict or lax. Issue #295") {
        val countExpectedErrors = (_: NonEmptyList[String]).count(_.contains("locus timepoint(s) in imaging sequence and not found in locus grouping"))
        
        forAll (Table(
            ("subfolder", "filename", "expectError"), 
            ("LocusGroupingValidation", "example__imaging_rounds_configuration__selective_prohibition__no_check_locus_grouping_295.json", false), 
            ("LocusGroupingValidation", "fail__imaging_rounds_config__missing_locus_validation_295.json", true),
            ("LocusGroupingValidation", "fail__imaging_rounds_config__null_locus_validation_295.json", true),
            ("LocusGroupingValidation", "fail__imaging_rounds_config__true_locus_validation_295.json", true),
        )) { (subfolder, filename, expectError) => 
            val configFile = getResourcePath(subfolder = subfolder, filename = filename)
            val safeParseResult = ImagingRoundsConfiguration.fromJsonFile(configFile)
            if (expectError) {
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
            ("LocusGroupingValidation", "example__rounds_config__with_times_correctly_omitted_from_locus_grouping__304.json", false), 
            ("LocusGroupingValidation", "fail__rounds_config__with_times_incorrectly_omitted_from_locus_grouping__304.json", true), 
        )) { (subfolder, filename, expectError) =>
            val configFile = getResourcePath(subfolder = subfolder, filename = filename)
            val safeParseResult = ImagingRoundsConfiguration.fromJsonFile(configFile)
            if (expectError) {
                safeParseResult match {
                    case Left(errorMessages) => 
                        println(s"ERRORS: ${errorMessages}")
                        errorMessages.count(_ === "2 locus timepoint(s) in imaging sequence and not found in locus grouping: 3, 4") shouldEqual 1
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

    private def getResourcePath(subfolder: String, filename: String): os.Path = 
        os.Path(getClass.getResource(s"/TestImagingRoundsConfiguration/$subfolder/$filename").getPath)

end TestImagingRoundsConfigurationExamplesParsability
