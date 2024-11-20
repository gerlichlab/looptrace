package at.ac.oeaw.imba.gerlich.looptrace

import squants.space.{ Length, Nanometers }
import org.scalacheck.*
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import at.ac.oeaw.imba.gerlich.looptrace.AssignTraceIds.{
    InputRecord, 
    workflow, 
}
import at.ac.oeaw.imba.gerlich.looptrace.space.Pixels3D

/** Tests for the program which assigns trace IDs to spots/ROIs */
class TestAssignTraceIds extends AnyFunSuite, LooptraceSuite, ScalaCheckPropertyChecks, should.Matchers:
    given Arbitrary[Pixels3D] = ???
    given Arbitrary[InputRecord] = ???

    test("Imaging rounds configuration with no merge rules simply emits records with all empty trace partners."):
        pending

    test("Imaging rounds configuration with no merge rules assigns correct trace IDs."):
        pending

    test("ROIs from timepoints not in merge rules are discarded if and only if the discard setting is active."):
        pending
    
    test("Absence of nucleus column is unproblematic."):
        pending
end TestAssignTraceIds
