package at.ac.oeaw.imba.gerlich.looptrace

import cats.data.*
import cats.syntax.all.*
import org.scalacheck.*
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2
import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2.given
import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2.syntax.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanDistance
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.testing.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.{
    ProximityGroup,
    RoiPartnersRequirementType,
    TraceIdDefinitionAndFiltrationRulesSet,
}
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.TraceIdDefinitionAndFiltrationRule
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeReal
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint

/** Tests for the parsing of the section about merging ROIs from different timepoints */
class TestParseDifferentTimepointsRoiMergeSectionOfImagingRoundsConfig extends 
    AnyFunSuite, 
    ScalaCheckPropertyChecks, 
    LooptraceSuite,
    ImagingRoundsConfigurationSuite, 
    should.Matchers:
    test("Parse succeeds when all parts are present and correctly structured"):
        import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
        import at.ac.oeaw.imba.gerlich.looptrace.instances.regionId.given

        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        def genGroup(using arbTime: Arbitrary[ImagingTimepoint]): Gen[AtLeast2[Set, ImagingTimepoint]] =
            Gen.choose(2, 10)
                .flatMap(n => Gen.containerOfN[Set, ImagingTimepoint](n, arbTime.arbitrary))
                .suchThat(_.size >= 2)
                .map{ g => 
                    AtLeast2.either(g).fold(
                        msg => throw new Exception(s"Error building group for test: $msg"),
                        identity
                    )
                }

        def genGroupingOfOne(using Arbitrary[ImagingTimepoint]): Gen[NonEmptyList[AtLeast2[Set, ImagingTimepoint]]] = 
            genGroup.map(NonEmptyList.one)
        
        def genGroupingOfTwo: Gen[NonEmptyList[AtLeast2[Set, ImagingTimepoint]]] = 
            val indices = (0 to 10).map(ImagingTimepoint.unsafeLift)
            for {
                ts1 <- Gen.choose(2, 5)
                    .flatMap(n => Gen.pick(n, indices))
                    .map(g => AtLeast2.unsafe(g.toSet))
                ts2 <- Gen.choose(2, indices.size - ts1.size)
                    .flatMap(n => Gen.pick(n, indices.toSet -- ts1.toSet))
                    .map(g => AtLeast2.unsafe(g.toSet))
            } yield NonEmptyList.of(ts1, ts2)

        given arbGrouping(using Arbitrary[ImagingTimepoint]): Arbitrary[NonEmptyList[AtLeast2[Set, ImagingTimepoint]]] = 
            Arbitrary{ Gen.oneOf(genGroupingOfOne, genGroupingOfTwo) }
            
        forAll: (threshold: EuclideanDistance.Threshold, requirement: RoiPartnersRequirementType, groups: NonEmptyList[AtLeast2[Set, ImagingTimepoint]], strict: Boolean) => 
            val groupsJson = groups.map(ids => "[" ++ ids.toList.map(_.show_).mkString(", ") ++ "]").mkString_(", ")
            val json = s"""{"discardRoisNotInGroupsOfInterest": $strict, "distanceThreshold": ${threshold.get.show_}, "requirementType": "${requirement}", "groups": [${groupsJson}]}"""
            TraceIdDefinitionAndFiltrationRulesSet.fromJson(json) match {
                case Left(messages) => 
                    fail(s"${messages.length} error(s) decoding JSON: ${messages.mkString_("; ")}" ++ "\n" ++ json)
                case Right(parsedResult) => 
                    val expGroups = groups.map{ g => TraceIdDefinitionAndFiltrationRule(ProximityGroup(threshold, g), requirement) }
                    parsedResult shouldEqual (expGroups, strict)
            }
    
    test("Simple single-group example parses as expected"):
        val strict = true
        val json = s"""{"discardRoisNotInGroupsOfInterest": $strict, "distanceThreshold": 5, "requirementType": "Conjunctive", "groups": [[9, 10]]}"""
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(json) match {
            case Left(messages) => 
                fail(s"${messages.length} error(s) decoding JSON: ${messages.mkString_("; ")}" ++ "\n" ++ json)
            case Right(parsedResult) => 
                import io.github.iltotore.iron.autoRefine
                val expectation = TraceIdDefinitionAndFiltrationRule(
                    ProximityGroup(
                        EuclideanDistance.Threshold(5: NonnegativeReal), 
                        AtLeast2.unsafe(Set(9, 10).map(ImagingTimepoint.unsafeLift))
                    ), 
                    RoiPartnersRequirementType.Conjunctive,
                )
                parsedResult shouldEqual (NonEmptyList.one(expectation), strict)
        }
    
    test("Simple two-group example parses as expected"):
        val threshold = 
            import io.github.iltotore.iron.autoRefine
            EuclideanDistance.Threshold(5: NonnegativeReal)
        val strict = true
        val json = s"""{"discardRoisNotInGroupsOfInterest": $strict, "distanceThreshold": ${threshold.get.show_}, "requirementType": "Conjunctive", "groups": [[1, 0], [9, 10]]}"""
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(json) match {
            case Left(messages) => 
                fail(s"${messages.length} error(s) decoding JSON: ${messages.mkString_("; ")}" ++ "\n" ++ json)
            case Right(parsedResult) => 
                val exp1 = TraceIdDefinitionAndFiltrationRule(
                    ProximityGroup(threshold, AtLeast2.unsafe(Set(1, 0).map(ImagingTimepoint.unsafeLift))), 
                    RoiPartnersRequirementType.Conjunctive,
                )
                val exp2 = TraceIdDefinitionAndFiltrationRule(
                    ProximityGroup(threshold, AtLeast2.unsafe(Set(9, 10).map(ImagingTimepoint.unsafeLift))),
                    RoiPartnersRequirementType.Conjunctive,
                )
                parsedResult shouldEqual (NonEmptyList.of(exp1, exp2), strict)
        }

    test("Without requirement type, no parse"):
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(
            s"""{"discardRoisNotInGroupsOfInterest": true, "distanceThreshold": 5, "groups": [[9, 10]]}"""
        ) match {
            case Left(messages) => 
                messages.count(_.contains("Missing requirement type for ROI merge")) shouldEqual 1
                messages.length shouldEqual 1
            case Right(_) => fail("Expected parse failure but got success")
        }
    
    test("Without threshold, no parse"):
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(
            s"""{"discardRoisNotInGroupsOfInterest": true, "requirementType": "Conjunctive", "groups": [[9, 10]]}"""
        ) match {
            case Left(messages) => 
                messages.count(_.contains("Missing threshold for ROI merge")) shouldEqual 1
                messages.length shouldEqual 1
            case Right(_) => fail("Expected parse failure but got success")
        }
    
    test("Threshold must be nonnegative"):
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(
            s"""{"discardRoisNotInGroupsOfInterest": true, "distanceThreshold": -1, "requirementType": "Conjunctive", "groups": [[9, 10]]}"""
        ) match {
            case Left(messages) => 
                messages.count(_.contains("!(Should be strictly negative)")) shouldEqual 1
                messages.length shouldEqual 1
            case Right(_) => fail("Expected parse failure but got success")
        }
    
    test("Each grouping member must have at least two members"):
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(
            s"""{"discardRoisNotInGroupsOfInterest": true, "distanceThreshold": 5, "requirementType": "Conjunctive", "groups": [[9, 10], [0]]}"""
        ) match {
            case Left(messages) => 
                messages.count(_.contains("Should have a minimum length of 2")) shouldEqual 1
                messages.length shouldEqual 1
            case Right(_) => fail("Expected parse failure but got success")
        }
    
    test("No member may be repeated within a group"):
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(
            s"""{"discardRoisNotInGroupsOfInterest": true, "distanceThreshold": 5, "requirementType": "Conjunctive", "groups": [[9, 10, 9], [0, 1]]}"""
        ) match {
            case Left(messages) => 
                messages.count(_.contains("2 unique value(s), but 3 total")) shouldEqual 1
                messages.length shouldEqual 1
            case Right(_) => fail("Expected parse failure but got success")
        }
    
    test("No member may be repeated between groups"):
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(
            s"""{"discardRoisNotInGroupsOfInterest": true, "distanceThreshold": 5, "requirementType": "Conjunctive", "groups": [[9, 10], [0, 9, 1]]}"""
        ) match {
            case Left(messages) => 
                messages.count(_.contains("repeated item(s) in merge rules")) shouldEqual 1
                messages.length shouldEqual 1
            case Right(_) => fail("Expected parse failure but got success")
        }
    
    test("Requirement type must be one of the fixed values"):
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(
            s"""{"discardRoisNotInGroupsOfInterest": true, "distanceThreshold": 5, "requirementType": "NotARealValue", "groups": [[9, 10]]}"""
        ) match {
            case Left(messages) => 
                messages.count(_.contains("Can't parse value for 'requirementType': NotARealValue")) shouldEqual 1
                messages.length shouldEqual 1
            case Right(_) => fail("Expected parse failure but got success")
        }
    
    test("With neither threshold nor requirement type, no parse"):
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(
            s"""{"discardRoisNotInGroupsOfInterest": true, "groups": [[9, 10]]}"""
        ) match {
            case Left(messages) => 
                messages.count(_.contains("Missing threshold and requirement type for ROI merge")) shouldEqual 1
                messages.length shouldEqual 1
            case Right(_) => fail("Expected parse failure but got success")
        }
    
    test("With no groups, no parse"):
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(
            s"""{"discardRoisNotInGroupsOfInterest": true, "distanceThreshold": 5, "requirementType": "Conjunctive"}"""
        ) match {
            case Left(messages) => 
                messages.count(_.contains("Missing key for groups: groups"))
                messages.length shouldEqual 1
            case Right(_) => fail("Expected parse failure but got success")
        }

    test("With empty groups, no parse"):
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(
            s"""{"discardRoisNotInGroupsOfInterest": true, "distanceThreshold": 5, "requirementType": "Conjunctive", "groups": []}"""
        ) match {
            case Left(messages) => 
                messages.count(_.contains(s"Groups (from 'groups') is empty")) shouldEqual 1
                messages.length shouldEqual 1
            case Right(_) => fail("Expected parse failure but got success")
        }

    test("With null groups, no parse"):
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(
            s"""{"discardRoisNotInGroupsOfInterest": true, "distanceThreshold": 5, "requirementType": "Conjunctive", "groups": null}"""
        ) match {
            case Left(messages) =>
                messages.count(_.contains(s"Can't parse groups (from 'groups') as array-like")) shouldEqual 1
                messages.length shouldEqual 1
            case Right(_) => fail("Expected parse failure but got success")
        }

    test("Without specification of discard policy, parse fails"):
        TraceIdDefinitionAndFiltrationRulesSet.fromJson(
            """{"distanceThreshold": 5, "requirementType": "Conjunctive", "groups": [[9, 10]]}"""
        ) match {
            case Left(messages) =>
                messages.count(_.contains("Missing key: 'discardRoisNotInGroupsOfInterest'")) shouldEqual 1
                messages.length shouldEqual 1
            case Right(_) => fail("Expected parse failure but got success")
        }

    test("With no groups and no threhsold, no parse"):
        pending
    
    test("With no groups and no requirement type, no parse"):
        pending
    
    test("With no groups and no threhsold and no requirement type, no parse"):
        pending
    
    test("Groups may not be all mappings."):
        pending
    
    test("No group may be a mapping."):
        pending
end TestParseDifferentTimepointsRoiMergeSectionOfImagingRoundsConfig
