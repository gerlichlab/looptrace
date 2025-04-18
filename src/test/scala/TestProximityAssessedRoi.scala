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
import at.ac.oeaw.imba.gerlich.looptrace.roi.{
    DetectedSpotRoi,
    MergerAssessedRoi,
}
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.IndexedDetectedSpot

/** Tests for the correctness of the main smart constructor of [[at.ac.oeaw.imba.gerlich.looptrace.MergerAssessedRoi]]. */
class TestMergerAssessedRoi extends AnyFunSuite, LooptraceSuite, ScalaCheckPropertyChecks, should.Matchers:
    test("MergerAssessedRoi.build correctly passes through the proximal ROI indices.") {
        // Generate legal combination of main ROI index, too-close ROIs, and ROIs to merge.
        given (Arbitrary[RoiIndex]) => Arbitrary[(RoiIndex, Set[RoiIndex])] = Arbitrary:
            for
                idx <- Arbitrary.arbitrary[RoiIndex]
                raw <- Gen.listOf(Arbitrary.arbitrary[RoiIndex])
            yield (idx, raw.toSet - idx)

        // Avoid shrinking so that the invariant about the main index being in the set of 
        // ROI indices "too close together" remains respected.
        given [A] => Shrink[A] = Shrink.shrinkAny

        forAll { (indexAndMerge: (RoiIndex, Set[RoiIndex]), spot: IndexedDetectedSpot) => 
            val (index, forMerge) = indexAndMerge
            MergerAssessedRoi.build(spot.copy(index = index), forMerge) match {
                case Left(messages) => 
                    fail(s"Expected ROI build success, but it failed with message(s): $messages")
                case Right(roi) => roi.mergeNeighbors shouldEqual forMerge
            }
        }
    }
    
    test("MergerAssessedRoi.build correctly enforces exclusion of ROI index from for-merge ROIs.") {
        // Generate ROI index in the collection of too-close ROIs, to trigger expected error.
        given (Arbitrary[RoiIndex]) => Arbitrary[(RoiIndex, Set[RoiIndex])] = Arbitrary{
            for
                idx <- Arbitrary.arbitrary[RoiIndex]
                raw <- Gen.listOf(Arbitrary.arbitrary[RoiIndex])
            yield (idx, raw.toSet + idx)
        }

        // Avoid shrinking so that the invariant about the main index being in the set of 
        // ROI indices "too close together" remains respected.
        given [A] => Shrink[A] = Shrink.shrinkAny

        forAll { (indexAndMerge: (RoiIndex, Set[RoiIndex]), spot: IndexedDetectedSpot) => 
            val (index, forMerge) = indexAndMerge
            MergerAssessedRoi.build(spot.copy(index = index), forMerge) match {
                case Left(errMsg) => errMsg.contains("An ROI cannot be merged with itself") shouldBe true
                case Right(_) => fail("Expected the ROI build to fail but it succeeded")
            }
        }
    }
end TestMergerAssessedRoi
