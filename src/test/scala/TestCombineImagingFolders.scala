package at.ac.oeaw.imba.gerlich.looptrace

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*

/** Tests for the combining of imaging folders (e.g. subsequences of imaging timepoints) */
class TestCombineImagingFolders extends AnyFunSuite, ScalacheckGenericExtras, ScalacheckSuite, should.Matchers:
    test("Total number of files doesn't change") { pending }
    test("No file contents change.") { pending }
    test("All input folders other than the output folder are empty at the end.") { pending }
    test("General correctness") { pending }
    test("Files without the desired extension are ignored.") { pending }
    test("Excution flag works.") { pending }
    test("Script is always (regardless of execution flag) written and is correct.") { pending }
end TestCombineImagingFolders
