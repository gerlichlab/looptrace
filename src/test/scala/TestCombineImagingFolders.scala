package at.ac.oeaw.imba.gerlich.looptrace

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Tests for the combining of imaging folders (e.g. subsequences of imaging timepoints) */
class TestCombineImagingFolders extends AnyFunSuite, ScalaCheckPropertyChecks, LooptraceSuite, should.Matchers:
    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)

    test("Don't regress in file name generation: #289") {
        val oldTime = ImagingTimepoint(NonnegativeInt(42))
        val oldTimeText = ImagingTimepoint.printForFilename(oldTime)
        val oldName = s"20240313_155456_993__${oldTimeText}_Point0012_ChannelFar Red,Red_Seq0558.nd2"
        val infile = os.pwd / oldName
        val targetFolder = os.pwd
        val filenameSep = "_"
        forAll { (newTime: ImagingTimepoint) => 
            CombineImagingFolders.makeSrcDstPair(targetFolder, filenameSep)(newTime, infile) match {
                case Left(error) => fail(s"${error.getMessage}")
                case Right((oldPath, newPath)) => 
                    oldPath shouldEqual infile
                    val newTimeText = ImagingTimepoint.printForFilename(newTime)
                    val newName = oldName.replace(oldTimeText, newTimeText)
                    val expPath = targetFolder / newName
                    newPath shouldEqual expPath
            }
        }
    }

    test("Total number of files doesn't change") { pending }
    test("No file contents change.") { pending }
    test("All input folders other than the output folder are empty at the end.") { pending }
    test("General correctness") { pending }
    test("Files without the desired extension are ignored.") { pending }
    test("Excution flag works.") { pending }
    test("Script is always (regardless of execution flag) written and is correct.") { pending }
end TestCombineImagingFolders
