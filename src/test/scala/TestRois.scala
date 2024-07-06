package at.ac.oeaw.imba.gerlich.looptrace

import upickle.default.*

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.looptrace.space.CoordinateSequence

/** Tests for the partitioning of regions of interest (ROIs) for drift correction */
class TestRois extends AnyFunSuite, ScalaCheckPropertyChecks, should.Matchers, PartitionRoisSuite {
    import SelectedRoi.*
    
    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)
    
    test("ROI for shifting roundtrips through JSON.") {
        forAll { (original: RoiForShifting, coordseq: CoordinateSequence) =>
            given rw: ReadWriter[RoiForShifting] = SelectedRoi.simpleShiftingRW(coordseq)
            val json = write(original)
            original shouldEqual read[RoiForShifting](json)
        }
    }

    test("ROI for accuracy roundtrips through JSON.") {
        forAll { (original: RoiForAccuracy, coordseq: CoordinateSequence) =>
            given rw: ReadWriter[RoiForAccuracy] = SelectedRoi.simpleAccuracyRW(coordseq)
            val json = write(original)
            original shouldEqual read[RoiForAccuracy](json)
        }
    }
}

