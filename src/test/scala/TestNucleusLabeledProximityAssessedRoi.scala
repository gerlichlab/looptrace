package at.ac.oeaw.imba.gerlich.looptrace

import cats.data.NonEmptyList
import cats.syntax.all.*

import org.scalacheck.*
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import io.github.iltotore.iron.scalacheck.all.given

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.collections.*
import at.ac.oeaw.imba.gerlich.gerlib.testing.instances.all.given

/** Tests for the correctness of the main smart constructor of [[at.ac.oeaw.imba.gerlich.looptrace.NucleusLabeledProximityAssessedRoi]]. */
class TestNucleusLabeledProximityAssessedRoi extends AnyFunSuite, LooptraceSuite, ScalaCheckPropertyChecks, should.Matchers:
    
    // The ROIs designated as too close together, and the ROIs to merge
    private type RoiBags = (Set[RoiIndex], Set[RoiIndex])

    test("NucleusLabeledProximityAssessedRoi.build correctly passes through the proximal ROI indices.") {
        pending
    }
    
    test("NucleusLabeledProximityAssessedRoi.build correctly prohibits any intersection between ROI sets.") {
        // Generate overlapping ROI index sets, to trigger the expected failure.
        given arbRoiBags(using Arbitrary[RoiIndex]): Arbitrary[RoiBags] = Arbitrary{
            for {
                raw1 <- Gen.nonEmptyListOf[RoiIndex](Arbitrary.arbitrary[RoiIndex])
                raw2Base <- Gen.nonEmptyListOf(Gen.oneOf(raw1))
                raw2Extra <- Arbitrary.arbitrary[List[RoiIndex]]
            } yield (raw1.toSet, (raw2Base ::: raw2Extra).toSet)
        }

        forAll { (index: RoiIndex, roi: DetectedSpotRoi, nucleus: NuclearDesignation, roiBags: RoiBags) => 
            whenever(roiBags._1.excludes(index) && roiBags._2.excludes(index)){
                NucleusLabeledProximityAssessedRoi.build(index, roi, nucleus, roiBags._1, roiBags._2) match {
                    case Left(messages) => 
                        val expMsg = "Overlap between too-close ROIs and ROIs to merge"
                        messages.count(_.contains(expMsg)) shouldEqual 1
                    case Right(_) => fail("Expected the ROI build to fail but it succeeded")
                }
            }
        }
    }
    
    test("NucleusLabeledProximityAssessedRoi.build correctly enforces exclusion of ROI index from too-close ROIs.") {
        // Generate ROI index in the collection of too-close ROIs, to trigger expected error.
        given arbRoiIndexAndRoiBags(using Arbitrary[RoiIndex]): Arbitrary[(RoiIndex, RoiBags)] = Arbitrary{
            for {
                idx <- Arbitrary.arbitrary[RoiIndex]
                raw1 <- Gen.listOf(Arbitrary.arbitrary[RoiIndex])
                bag1 = raw1.toSet + idx
                raw2 <- Gen.listOf(Arbitrary.arbitrary[RoiIndex])
                bag2 = raw2.toSet -- bag1 // Prevent overlap with other bag and with main index.
            } yield (idx, bag1 -> bag2)
        }

        // Avoid shrinking so that the invariant about the main index being in the set of 
        // ROI indices "too close together" remains respected.
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny

        forAll { (indexAndBags: (RoiIndex, RoiBags), roi: DetectedSpotRoi, nucleus: NuclearDesignation) => 
            val (index, (tooClose, forMerge)) = indexAndBags
            NucleusLabeledProximityAssessedRoi.build(index, roi, nucleus, tooClose, forMerge) match {
                case Left(messages) => 
                    val expMsg = "An ROI cannot be too close to itself"
                    messages.count(_.contains(expMsg)) shouldEqual 1
                case Right(_) => fail("Expected the ROI build to fail but it succeeded")
            }
        }
    }
    
    test("NucleusLabeledProximityAssessedRoi.build correctly enforces exclusion of ROI index from ROIs to merge.") {
        // Generate ROI index in the collection of ROIs to merge, to trigger expected error.
        given arbRoiIndexAndRoiBags(using Arbitrary[RoiIndex]): Arbitrary[(RoiIndex, RoiBags)] = Arbitrary{
            for {
                idx <- Arbitrary.arbitrary[RoiIndex]
                raw2 <- Gen.listOf(Arbitrary.arbitrary[RoiIndex])
                bag2 = raw2.toSet + idx
                raw1 <- Gen.listOf(Arbitrary.arbitrary[RoiIndex])
                bag1 = raw1.toSet -- bag2 // Prevent overlap with other bag and with main index.
            } yield (idx, bag1 -> bag2)
        }

        // Avoid shrinking so that the invariant about the main index being in the set of 
        // ROI indices "too close together" remains respected.
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny

        forAll { (indexAndBags: (RoiIndex, RoiBags), roi: DetectedSpotRoi, nucleus: NuclearDesignation) => 
            val (index, (tooClose, forMerge)) = indexAndBags
            NucleusLabeledProximityAssessedRoi.build(index, roi, nucleus, tooClose, forMerge) match {
                case Left(messages) => 
                    val expMsg = "An ROI cannot be merged with itself"
                    messages.count(_.contains(expMsg)) shouldEqual 1
                case Right(_) => fail("Expected the ROI build to fail but it succeeded")
            }
        }
    }
end TestNucleusLabeledProximityAssessedRoi
