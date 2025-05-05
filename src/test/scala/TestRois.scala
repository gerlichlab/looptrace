package at.ac.oeaw.imba.gerlich.looptrace

import upickle.default.*

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

/** Tests for the partitioning of regions of interest (ROIs) for drift
  * correction
  */
class TestRois
    extends AnyFunSuite,
      ScalaCheckPropertyChecks,
      should.Matchers,
      PartitionRoisSuite {
  import SelectedRoi.*

  override implicit val generatorDrivenConfig: PropertyCheckConfiguration =
    PropertyCheckConfiguration(minSuccessful = 100)

  test("ROI for shifting roundtrips through JSON.") {
    forAll { (original: RoiForShifting) =>
      given rw: ReadWriter[RoiForShifting] = SelectedRoi.simpleShiftingRW
      val json = write(original)
      original shouldEqual read[RoiForShifting](json)
    }
  }

  test("ROI for accuracy roundtrips through JSON.") {
    forAll { (original: RoiForAccuracy) =>
      given rw: ReadWriter[RoiForAccuracy] = SelectedRoi.simpleAccuracyRW
      val json = write(original)
      original shouldEqual read[RoiForAccuracy](json)
    }
  }
}
