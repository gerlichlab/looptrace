package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ Random, Try }

import cats.Order
import cats.data.{ NonEmptyList, NonEmptySet }
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.imaging.{
    FieldOfView,
    ImagingTimepoint,
}
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.instances.simpleShow.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.testing.instances.NumericInstances
import at.ac.oeaw.imba.gerlich.gerlib.testing.syntax.SyntaxForScalacheck

import at.ac.oeaw.imba.gerlich.looptrace.PartitionIndexedDriftCorrectionBeadRois.*
import at.ac.oeaw.imba.gerlich.looptrace.PathHelpers.listPath
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.readJsonFile
import at.ac.oeaw.imba.gerlich.looptrace.space.{ Point3D, XCoordinate, YCoordinate, ZCoordinate }
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Tests for the partitioning of regions of interest (ROIs) for drift correction */
class TestPartitionIndexedDriftCorrectionBeadRois extends 
    AnyFunSuite, 
    NumericInstances, 
    ScalaCheckPropertyChecks, 
    SyntaxForScalacheck, 
    should.Matchers, 
    PartitionRoisSuite:
    
    import SelectedRoi.*

    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)
    
    test("Partition.RepeatedRoisWithinPartError is accessible but cannot be directly built.") {
        /* Make the lines shorter by aliasing these constructors. */
        import RoisSplit.Partition.RepeatedRoisWithinPartError as RepRoisError
        import XCoordinate.apply as x, YCoordinate.apply as y, ZCoordinate.apply as z

        /* Certainly not buildable with empty mappings */
        assertCompiles("import RepRoisError.apply") // accessibility, negative control
        assertTypeError("RepRoisError(Map(), Map())")
        
        /* Even fails with the value restriction (> 1) satisfied, since the constructor's private. */
        assertCompiles("RoiForShifting(RoiIndex(NonnegativeInt(0)), Point3D(x(1.0), y(0.0), z(2.0)))") // Ensure the first snippet to use is error-free.
        assertCompiles("RoiForShifting(RoiIndex(NonnegativeInt(1)), Point3D(x(0.0), y(-1.0), z(1.0)))") // Ensure the other snippet to use is error-free.
        assertTypeError("RepRoisError(Map(RoiForShifting(RoiIndex(NonnegativeInt(0)), Point3D(x(1.0), y(0.0), z(2.0))) -> 2), Map(RoiForShifting(RoiIndex(NonnegativeInt(1)), Point3D(x(0.0), y(-1.0), z(1.0))) -> 2))")
    }

    test("ShiftingCount appropriately constrains the domain.") {
        assertDoesNotCompile("ShiftingCount(9)")
        assertCompiles("ShiftingCount(10)")
        intercept[IllegalArgumentException]{ 
            ShiftingCount.unsafe(ShiftingCount.AbsoluteMinimumShifting - 1)
        }.getMessage shouldEqual s"Insufficient value (< ${ShiftingCount.AbsoluteMinimumShifting}) for shifting count!"
        ShiftingCount.unsafe(ShiftingCount.AbsoluteMinimumShifting : Int) shouldBe ShiftingCount(10)
    }

    test("RoisSplit.TooFewShifting requires ShiftingCount, NonnegativeInt, and PositiveInt.") {
        /* "Negative" (no error) case as control */
        assertCompiles("RoisSplit.TooFewShifting(ShiftingCount(100), NonnegativeInt(100), PositiveInt(100))")
        /* Relaxing the first argument */
        assertTypeError("RoisSplit.TooFewShifting(PositiveInt(100), NonnegativeInt(100), PositiveInt(100))")
        assertTypeError("RoisSplit.TooFewShifting(NonnegativeInt(100), NonnegativeInt(100), PositiveInt(100))")
        assertTypeError("RoisSplit.TooFewShifting(100, NonnegativeInt(100), PositiveInt(100))")
        /* Relaxing the second argument */
        assertTypeError("RoisSplit.TooFewShifting(ShiftingCount(100), 100, PositiveInt(100))")
        /* Relaxing the third argument */
        assertTypeError("RoisSplit.TooFewShifting(ShiftingCount(100), NonnegativeInt(100), Nonnegative(100))")
        assertTypeError("RoisSplit.TooFewShifting(ShiftingCount(100), NonnegativeInt(100), 100)")
        /* Permuting the arguments */
        assertTypeError("RoisSplit.TooFewShifting(ShiftingCount(100), PositiveInt(100), NonnegativeInt(100))")
        assertTypeError("RoisSplit.TooFewShifting(NonnegativeInt(100), ShiftingCount(100), PositiveInt(100))")
        assertTypeError("RoisSplit.TooFewShifting(NonnegativeInt(100), PositiveInt(100), ShiftingCount(100))")
        assertTypeError("RoisSplit.TooFewShifting(PositiveInt(100), NonnegativeInt(100), ShiftingCount(100))")
        assertTypeError("RoisSplit.TooFewShifting(PositiveInt(100), ShiftingCount(100), NonnegativeInt(100))")
    }

    test("Attempt to partition ROIs when there are repeats in parts fails expectedly.") {
        given [A] => Shrink[A] = Shrink.shrinkAny[A]
        type RoisAndReps[R] = (List[R], Map[R, Int])
        def genRoisAndReps[R](base: List[R]): Gen[RoisAndReps[R]] = 
            if base.isEmpty then List() -> Map()
            // Add 1 to each count to represent the value being duplicated.
            else Gen.resize(5, Gen.listOf(Gen.oneOf(base))).fproduct(_.groupBy(identity).view.mapValues(_.length + 1).toMap)
        def genWithRepeats: Gen[(RoisAndReps[RoiForShifting], RoisAndReps[RoiForAccuracy])] = {
            val maxNumRois = 50
            for
                // NB: relying on randomness of Point3D and zero-probability of collision there to mitigate risk that repeats 
                //     are generated in the baseX collections, which would throw off the counting of expected repeats.
                baseShifting <- Gen.choose(ShiftingCount.AbsoluteMinimumShifting, maxNumRois).flatMap(Gen.listOfN(_, arbitrary[RoiForShifting]))
                baseAccuracy <- Gen.resize(maxNumRois - baseShifting.length, Gen.listOf(arbitrary[RoiForAccuracy]))
                (shifting, accuracy) <- (genRoisAndReps(baseShifting), genRoisAndReps(baseAccuracy))
                    .tupled
                    .suchThat((del, acc) => del._2.nonEmpty || acc._2.nonEmpty)
                    .map{ case ((repDel, expDel), (repAcc, expAcc)) => (
                        Random.shuffle(repDel ::: baseShifting).toList -> expDel, 
                        Random.shuffle(repAcc ::: baseAccuracy).toList -> expAcc
                        )
                    }
            yield (shifting, accuracy)
        }
        def genNumShift = Gen.choose(ShiftingCount.AbsoluteMinimumShifting, Int.MaxValue).map(ShiftingCount.unsafe)

        forAll (genWithRepeats, genNumShift, arbitrary[PositiveInt]) { 
            case (((shiftingRois, expShiftingReps), (accuracyRois, expAccuracyReps)), numShifting, numAccuracy) => 
                val error = intercept[RoisSplit.Partition.RepeatedRoisWithinPartError]{
                    RoisSplit.Partition.build(numShifting, shiftingRois, numAccuracy, accuracyRois)
                }
                error.shifting shouldEqual expShiftingReps
                error.accuracy shouldEqual expAccuracyReps
        }
    }

    test("Partition's constructor is private, but the result types' constructors are public") {
        assertCompiles("RoisSplit.TooFewShifting.apply")
        assertCompiles("RoisSplit.TooFewAccuracyRescued.apply")
        assertCompiles("RoisSplit.TooFewAccuracyHealthy.apply")
        assertTypeError("RoisSplit.Partition.apply")
    }

    test("ROI request sizes must be correct integer subtypes.") {
        assertCompiles("sampleDetectedRois(ShiftingCount(10), PositiveInt(10))(List())") // negative control
        
        /* Alternatives still using ShiftingCount */
        assertTypeError("sampleDetectedRois(ShiftingCount(10), NonnegativeInt(10))(List())")
        assertTypeError("sampleDetectedRois(ShiftingCount(10), 10)(List())")

        /* Alternatives with at least 1 positive int */
        assertTypeError("sampleDetectedRois(PositiveInt(10), PositiveInt(10))(List())")
        assertTypeError("sampleDetectedRois(PositiveInt(10), NonnegativeInt(10))(List())")
        assertTypeError("sampleDetectedRois(NonnegativeInt(10), PositiveInt(10))(List())")
        assertTypeError("sampleDetectedRois(PositiveInt(10), 10)(List())")
        assertTypeError("sampleDetectedRois(10, PositiveInt(10))(List())")
        
        /* Other alternatives with at least 1 nonnegative int */
        assertTypeError("sampleDetectedRois(NonnegativeInt(10), NonnegativeInt(10))(List())")
        assertTypeError("sampleDetectedRois(NonnegativeInt(10), 10)(List())")
        assertTypeError("sampleDetectedRois(10, NonnegativeInt(10))(List())")
        
        // Alternative with simple integers
        assertTypeError("sampleDetectedRois(10, 10)(List())")
    }

    test("Entirely empty ROIs file (no header even) causes expected error.") {
        forAll { (delimiter: Delimiter) => 
            withTempFile("", delimiter){ (roisFile: os.Path) => 
                os.isFile(roisFile) shouldBe true
                val expErrorMessages = NonEmptyList.one(s"No lines in file! $roisFile")
                val expected = RoisFileParseFailedSetup(expErrorMessages).asLeft[Iterable[FiducialBead]]
                val observed = readRoisFile(roisFile)
                observed shouldEqual expected
            }
        }
    }

    test("Bad ROIs file extension causes expected error.") {
        given [A] => Shrink[A] = Shrink.shrinkAny[A]

        def genInvalidExt: Gen[String] = Gen.alphaNumStr.suchThat{ ext => Delimiter.fromExtension(ext).isEmpty }.map("." ++ _)
        def genHeaderAndGetExtraErrorOpt: Gen[(String, Option[os.Path => String])] = Gen.choose(0, 5).flatMap{
            case 0 => Gen.const(("", ((p: os.Path) => s"No lines in file! $p").some))
            case n => for
                delim <- arbitrary[Delimiter]
                fields <- Gen.listOfN(n, Gen.alphaNumStr.suchThat(_.nonEmpty))
            yield (delim.join(fields.toArray), None)
        }

        forAll (genInvalidExt, genHeaderAndGetExtraErrorOpt) { 
            case (ext, (header, maybeGetExpError)) => 
                withTempFile(initData = header, suffix = ext){ (roisFile: os.Path) => 
                    val extras: List[String] = maybeGetExpError.fold(List())(getMsg => List(getMsg(roisFile)))
                    val expErrorMessages = NonEmptyList(s"Cannot infer delimiter for file! $roisFile", extras)
                    readRoisFile(roisFile) shouldEqual Left(RoisFileParseFailedSetup(expErrorMessages))
                }
        }
    }

    test("Any missing column name in header causes error.") {
        given [A] => Shrink[A] = Shrink.shrinkAny
        
        // Create the parser config and a strict subset of the column names.
        def genHeadFieldSubset: Gen[List[String]] = 
            // Generate at least one value, to avoid potential conflict of thinking 
            // that the index column (perhaps empty string) is present.
            Gen.choose(1, ColumnNamesToParse.length - 1)
                .flatMap(Gen.pick(_, ColumnNamesToParse))
                .map(_.toList)
                // Don't generate just the empty string, as that could conceivably 
                // trigger an error that the file is empty.
                .suchThat(_ =!= List(""))

        // Optionally, generate some additional column names, limiting to relatively few columns.
        def genHeaderAndDelimiter = for
            headerSubset <- genHeadFieldSubset
            usefulColumns = ColumnNamesToParse.toSet
            genCol = Gen.alphaNumStr.suchThat(!usefulColumns.contains(_))
            extras <- Gen.choose(0, 5).flatMap(Gen.listOfN(_, genCol))
            delimiter <- arbitrary[Delimiter]
        yield (Random.shuffle(headerSubset ::: extras), delimiter)
        
        forAll (genHeaderAndDelimiter) { 
            case (headerFields, delimiter) =>
                val expMissFields = ColumnNamesToParse.toSet.toSet -- headerFields.toSet
                val expMessages = expMissFields.map(name => s"Missing field in header: $name")
                val headLine = delimiter.join(headerFields.toArray) ++ "\n"
                withTempFile(headLine, delimiter){ (roisFile: os.Path) => 
                    readRoisFile(roisFile) match {
                        case Right(_) => fail("ROIs file read succeeded when it should've failed!")
                        case Left(RoisFileParseFailedSetup(errorMessages)) => 
                            errorMessages.toList.toSet shouldEqual expMessages
                            errorMessages.length shouldEqual expMissFields.size
                        case Left(e) => fail(s"Parse failed but in unexpected (non-setup) way: $e")
                    }
                }
            }
    }

    test("ANY bad row fails the parse.") {
        /* Inter-field delimiter and header for the ROIs file */
        val delimiter = Delimiter.CommaSeparator
        val headLine = "beadIndex,label,zc,yc,xc,max_intensity,area,fail_code"

        // Pairs of ROIs file lines and corresponding expectation
        val inputAndExpPairs = Table(
            ("inputLines", "expBadRecords"),
            (
                headLine :: List(
                    "101,102,11.96875,1857.9375,1076.25,26799.0,32.0,", 
                    "104,105,10.6,1919.8,1137.4,12858.0,5.0,,"
                    ), 
                NonEmptyList.one(BadRecord(NonnegativeInt(1), delimiter.split("104,105,10.6,1919.8,1137.4,12858.0,5.0,,"), NonEmptyList.one("Header has 8 fields but record has 9")))
            ),
            (
                headLine :: List(
                    "101,102,11.96875,1857.9375,1076.25,26799.0,32.0,", 
                    "104,105,10.6,1919.8,1137.4,12858.0,5.0"
                    ), 
                NonEmptyList.one(BadRecord(NonnegativeInt(1), delimiter.split("104,105,10.6,1919.8,1137.4,12858.0,5.0"), NonEmptyList.one("Header has 8 fields but record has 7")))
            ), 
            (
                headLine :: List(
                    "101,102,11.96875,1857.9375,1076.25,26799.0,32.0", 
                    "104,105,10.6,1919.8,1137.4,12858.0,5.0,i", 
                    "109,107,11.96875,1857.9375,1076.25,26799.0"
                    ), 
                NonEmptyList(
                    BadRecord(NonnegativeInt(0), delimiter.split("101,102,11.96875,1857.9375,1076.25,26799.0,32.0"), NonEmptyList.one("Header has 8 fields but record has 7")), 
                    List(BadRecord(NonnegativeInt(2), delimiter.split("109,107,11.96875,1857.9375,1076.25,26799.0"), NonEmptyList.one("Header has 8 fields but record has 6")))
                )
            )
        )

        forAll (inputAndExpPairs) {
            case (inputLines, expBadRecords) => 
                withTempDirectory{ (tempdir: os.Path) =>
                    val roisFile = tempdir / s"rois.${delimiter.ext}"
                    os.write.over(roisFile, inputLines.map(_ ++ "\n"))
                    readRoisFile(roisFile) match
                        case Right(_) => fail("Parse succeeded when it should've failed!")
                        case Left(bads) => bads shouldEqual RoisFileParseFailedRecords(expBadRecords)
                }
        }
    }

    test("Input discovery works as expected for folder with no other contents.") {
        forAll (genDistinctNonnegativePairs) { case (pt1, pt2) => {
            withTempDirectory{ (p: os.Path) => 
                val expected = Set(pt1, pt2).map{ pt => pt -> (p / getInputFilename(pt._1, pt._2)) }

                /* Check that inputs don't already exist, then establish them and check existence. */
                val expPaths = expected.map(_._2)
                expPaths.exists(os.exists) shouldBe false
                expPaths.foreach(touchFile(_, false))
                expPaths.forall(os.isFile) shouldBe true

                val found = discoverInputs(p) // Perform the empirical action.
                found shouldEqual expected
            }
        } }
    }

    test("Input discovery works as expected for mixed folder contents.") {
        enum FolderChoice:
            case Root, GoodSubfolder, BadSubfolder
        def setup(root: os.Path, pt1: FovTimePair, pt2: FovTimePair, fc: FolderChoice): (os.Path, Set[InitFile]) = {
            import FolderChoice.*
            val subGood = root / "sub1"
            val subBad = root / "sub2"
            val (pos, time) = pt1
            val baseFilename = getInputFilename(pos, time)
            val wrongPrefixFile = subGood / baseFilename.replaceAll(BeadRoisPrefix, "BadPrefix")
            val wrongSubfolderFile = subBad / baseFilename
            val missingPrefixFile = subGood / baseFilename.replaceAll(BeadRoisPrefix, "")
            val wrongFilenameStructureFile1 = 
                subGood / baseFilename.replaceAll(s"${pos.show_}_${time.show_}", s"${pos.show_}.${time.show_}")
            val wrongFilenameStructureFile2 = 
                subGood / baseFilename.replaceAll(s"${pos.show_}_${time.show_}", s"${pos.show_}_${time.show_}_0")
            val goodFile1 = subGood / baseFilename
            val goodFile2 = subGood / getInputFilename(pt2._1, pt2._2)
            List(
                wrongPrefixFile, 
                wrongSubfolderFile, 
                missingPrefixFile, 
                wrongFilenameStructureFile1, 
                wrongFilenameStructureFile2, 
                goodFile1, 
                goodFile2
                ).foreach(touchFile(_, true))
            fc match {
                case Root => root -> Set()
                case GoodSubfolder => subGood -> Set(pt1 -> goodFile1, pt2 -> goodFile2)
                case BadSubfolder => subBad -> Set(pt1 -> wrongSubfolderFile)
            }
        }
        forAll (Gen.zip(genDistinctNonnegativePairs, Gen.oneOf(FolderChoice.values.toList))) { 
            case ((pt1, pt2), folderChoice) => 
                withTempDirectory({ (root: os.Path) => 
                    val (inputPath, expectedOutput) = setup(root, pt1, pt2, folderChoice)
                    
                    /* Pretest the folder structure */
                    os.list(root).filter(os.isDir).toSet shouldEqual Set("sub1", "sub2").map(root / _)
                    os.list(root).filter(os.isFile).toSet shouldEqual Set()
                    
                    discoverInputs(inputPath) shouldEqual expectedOutput // Test the actual behavior of interest.
                })
        }
    }

    test("Sampling result accords with expectation based on relation between usable ROI count and requested ROI counts."):
        pending

    test("Cases of TooFewHealthyRoisRescued are correct and generate expected (implied) too-few-accuracy-ROIs records."):
        pending

    test("A ROI is never used for more than one purpose."):
        pending

    test("ROI counts fewer than absolute minimum results in expected warnings."):
        pending
    
    test("Warnings file is correct and produced IF AND ONLY IF there is at least one case of too-few-ROIs."):
        pending

    test("When shifting request takes up all usable ROIs available, JSON files are still written for accuracy but are empty."):
        pending

    test("No unusable ROI is ever used, and ROI indices and coordiantes are preserved during partition."):
        pending
    
    /* *******************************************************************************
     * Ancillary types and functions
     * *******************************************************************************
     */
    type NNPair = (NonnegativeInt, NonnegativeInt)

    /** Minimal detected bead ROIs field consumed by the partitioning program under test */
    val ColumnNamesToParse = List(
        ParserConfig.indexCol, 
        ParserConfig.xCol.get, 
        ParserConfig.yCol.get, 
        ParserConfig.zCol.get, 
    )

    def getInputFilename = (p: FieldOfView, t: ImagingTimepoint) => BeadsFilenameDefinition(p, t).getFilteredInputFilename

    /** Generate a pair of pairs of nonnegative integers such that the first pair isn't the same as the second. */
    def genDistinctNonnegativePairs: Gen[(FovTimePair, FovTimePair)] = 
        Gen.zip(arbitrary[(NonnegativeInt, NonnegativeInt)], arbitrary[(NonnegativeInt, NonnegativeInt)])
            .suchThat{ case (p1, p2) => p1 =!= p2 }
            .map { case ((p1, f1), (p2, f2)) => (FieldOfView(p1) -> ImagingTimepoint(f1), FieldOfView(p2) -> ImagingTimepoint(f2)) }

end TestPartitionIndexedDriftCorrectionBeadRois
