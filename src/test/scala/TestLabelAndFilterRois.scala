package at.ac.oeaw.imba.gerlich.looptrace

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*

/** Tests for the filtration of the individual supports (single FISH probes) of chromatin fiber traces */
class TestLabelAndFilterRois extends AnyFunSuite, GenericSuite, ScalacheckSuite, ScalacheckGenericExtras, should.Matchers:

    test("ROIs parse correctly from CSV") { pending }

    test("Spot distance comparison requires drift correction") { pending }
    test("Spot distance comparison uses drift correction.") { pending }
    test("Spot distance comparison responds to change of proximity comparison strategy #146.") { pending }

    // This tests for both the ability to specify nothing for the grouping, and for the correctness of the definition of the partitioning (trivial) when no grouping is specified.
    test("Spot distance comparison considers all ROIs as one big group if no grouping is provided. #147") { pending }
    
    test("In each pair of proximal spots, BOTH are filtered. #148") { pending }
    
    test("Probe grouping must partition regional barcode frames: A probe grouping declaration that does not cover regional barcodes set is an error.") { pending }
    
    test("Probe grouping must partition regional barcode frames: A probe grouping declaration with an overlap (probe/frame repeated between groups) an error.") { pending }

    test("ROI grouping by frame/probe/timepoint is specific to field-of-view, so that ROIs from different FOVs don't affect each other for filtering.") { pending }
end TestLabelAndFilterRois
