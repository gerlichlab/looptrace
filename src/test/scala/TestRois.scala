package at.ac.oeaw.imba.gerlich.looptrace

import upickle.default.*

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.looptrace.space.CoordinateSequence

/** Tests for the partitioning of regions of interest (ROIs) for drift correction */
class TestRois extends AnyFunSuite with ScalaCheckPropertyChecks with should.Matchers with PartitionRoisSuite {
    import SelectedRoi.*

    test("ROI for shifting roundtrips through JSON") {
        forAll { (original: RoiForShifting, coordseq: CoordinateSequence) =>
            given rw: ReadWriter[RoiForShifting] = SelectedRoi.simpleShiftingRW(coordseq)
            val json = write(original)
            original shouldEqual read[RoiForShifting](json)
        }
    }

    test("ROI for accuracy roundtrips through JSON") {
        forAll { (original: RoiForAccuracy, coordseq: CoordinateSequence) =>
            given rw: ReadWriter[RoiForAccuracy] = SelectedRoi.simpleAccuracyRW(coordseq)
            val json = write(original)
            original shouldEqual read[RoiForAccuracy](json)
        }
    }
}

