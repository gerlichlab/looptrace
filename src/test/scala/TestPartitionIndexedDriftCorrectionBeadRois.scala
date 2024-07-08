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

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given

import at.ac.oeaw.imba.gerlich.looptrace.PartitionIndexedDriftCorrectionBeadRois.*
import at.ac.oeaw.imba.gerlich.looptrace.PathHelpers.listPath
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.readJsonFile
import at.ac.oeaw.imba.gerlich.looptrace.space.{ CoordinateSequence, Point3D, XCoordinate, YCoordinate, ZCoordinate }
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Tests for the partitioning of regions of interest (ROIs) for drift correction */
class TestPartitionIndexedDriftCorrectionBeadRois extends AnyFunSuite, ScalaCheckPropertyChecks, ScalacheckGenericExtras, should.Matchers, PartitionRoisSuite:
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
        intercept[NumberFormatException]{ 
            ShiftingCount.unsafe(AbsoluteMinimumShifting - 1)
        }.getMessage shouldEqual s"Cannot use ${AbsoluteMinimumShifting - 1} as shifting count (min. $AbsoluteMinimumShifting)"
        ShiftingCount.unsafe(AbsoluteMinimumShifting : Int) shouldBe ShiftingCount(10)
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
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        type RoisAndReps[R] = (List[R], Map[R, Int])
        def genRoisAndReps[R](base: List[R]): Gen[RoisAndReps[R]] = 
            if base.isEmpty then List() -> Map()
            // Add 1 to each count to represent the value being duplicated.
            else Gen.resize(5, Gen.listOf(Gen.oneOf(base))).fproduct(_.groupBy(identity).view.mapValues(_.length + 1).toMap)
        def genWithRepeats: Gen[(RoisAndReps[RoiForShifting], RoisAndReps[RoiForAccuracy])] = {
            val maxNumRois = 50
            for {
                // NB: relying on randomness of Point3D and zero-probability of collision there to mitigate risk that repeats 
                //     are generated in the baseX collections, which would throw off the counting of expected repeats.
                baseShifting <- Gen.choose(AbsoluteMinimumShifting, maxNumRois).flatMap(Gen.listOfN(_, arbitrary[RoiForShifting]))
                baseAccuracy <- Gen.resize(maxNumRois - baseShifting.length, Gen.listOf(arbitrary[RoiForAccuracy]))
                (shifting, accuracy) <- (genRoisAndReps(baseShifting), genRoisAndReps(baseAccuracy))
                    .tupled
                    .suchThat((del, acc) => del._2.nonEmpty || acc._2.nonEmpty)
                    .map{ case ((repDel, expDel), (repAcc, expAcc)) => (
                        Random.shuffle(repDel ::: baseShifting).toList -> expDel, 
                        Random.shuffle(repAcc ::: baseAccuracy).toList -> expAcc
                        )
                    }
            } yield (shifting, accuracy)
        }
        def genNumShift = Gen.choose(AbsoluteMinimumShifting, Int.MaxValue).map(ShiftingCount.unsafe)

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
                readRoisFile(roisFile) shouldEqual Left(NonEmptyList.one(s"No lines in file! $roisFile"))
            }
        }
    }

    test("Bad ROIs file extension causes expected error.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        def genInvalidExt: Gen[String] = Gen.alphaNumStr.suchThat{ ext => Delimiter.fromExtension(ext).isEmpty }.map("." ++ _)
        def genHeaderAndGetExtraErrorOpt: Gen[(String, Option[os.Path => String])] = Gen.choose(0, 5).flatMap{
            case 0 => Gen.const(("", ((p: os.Path) => s"No lines in file! $p").some))
            case n => for {
                delim <- arbitrary[Delimiter]
                fields <- Gen.listOfN(n, Gen.alphaNumStr.suchThat(_.nonEmpty))
            } yield (delim.join(fields.toArray), None)
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
        // Create the parser config and a strict subset of the column names.
        def genHeadFieldSubset: Gen[List[String]] = 
            Gen.choose(0, ColumnNamesToParse.length - 1).flatMap(Gen.pick(_, ColumnNamesToParse)).map(_.toList)

        // Optionally, generate some additional column names, limiting to relatively few columns.
        def genHeaderAndDelimiter = for {
            headerSubset <- genHeadFieldSubset
            usefulColumns = ColumnNamesToParse.toSet
            genCol = Gen.alphaNumStr.suchThat(!usefulColumns.contains(_))
            extras <- Gen.choose(0, 5).flatMap(Gen.listOfN(_, genCol))
            delimiter <- arbitrary[Delimiter]
        } yield (Random.shuffle(headerSubset ::: extras), delimiter)
        
        forAll (genHeaderAndDelimiter) { 
            case (headerFields, delimiter) =>
                val expMissFields = ColumnNamesToParse.toSet.toSet -- headerFields.toSet
                val expMessages = expMissFields.map(name => s"Missing field in header: $name")
                val headLine = delimiter.join(headerFields.toArray) ++ "\n"
                withTempFile(headLine, delimiter){ (roisFile: os.Path) => 
                    readRoisFile(roisFile) match {
                        case Right(_) => fail("ROIs file read succeeded when it should've failed!")
                        case Left(RoisFileParseFailedSetup(errorMessages)) => 
                            errorMessages.length shouldEqual expMissFields.size
                            errorMessages.toList.toSet shouldEqual expMessages
                        case Left(e) => fail(s"Parse failed but in unexpected (non-setup) way: $e")
                    }
                }
            }
    }

    test("ANY bad row fails the parse.") {
        /* Inter-field delimiter and header for the ROIs file */
        val delimiter = Delimiter.CommaSeparator
        val headLine = ",label,centroid-0,centroid-1,centroid-2,max_intensity,area,fail_code"

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
                val expected = Set(pt1, pt2).map(pt => pt -> (p / getInputFilename(pt._1, pt._2)))

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
        def setup(root: os.Path, pt1: PosTimePair, pt2: PosTimePair, fc: FolderChoice): (os.Path, Set[InitFile]) = {
            import FolderChoice.*
            val subGood = root / "sub1"
            val subBad = root / "sub2"
            val (pos, time) = pt1
            val baseFilename = getInputFilename(pos, time)
            val wrongPrefixFile = subGood / baseFilename.replaceAll(BeadRoisPrefix, "BadPrefix")
            val wrongSubfolderFile = subBad / baseFilename
            val missingPrefixFile = subGood / baseFilename.replaceAll(BeadRoisPrefix, "")
            val wrongFilenameStructureFile1 = 
                subGood / baseFilename.replaceAll(s"${pos.get}_${time.get}", s"${pos.get}.${time.get}")
            val wrongFilenameStructureFile2 = 
                subGood / baseFilename.replaceAll(s"${pos.get}_${time.get}", s"${pos.get}_${time.get}_0")
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

    test("Sampling result accords with expectation based on relation between usable ROI count and requested ROI counts.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        type InputsAndValidate = (ShiftingCount, PositiveInt, List[DetectedRoi], RoisSplit.Result => Any)
        
        def genFewerThanAbsoluteMinimum: Gen[InputsAndValidate] = for {
            numShifting <- Gen.choose[Int](AbsoluteMinimumShifting, maxNumRoisSmallTests).map(ShiftingCount.unsafe)
            numAccuracy <- arbitrary[PositiveInt]
            (usable, unusable) <- (for {
                goods <- genUsableRois(0, AbsoluteMinimumShifting - 1)
                bads <- genUnusableRois(0, maxNumRoisSmallTests - goods.length)
            } yield (goods, bads)).suchThat{ (goods, bads) => // Ensure uniqueness among ROIs.
                (goods.toSet ++ bads.toSet).size === goods.size + bads.size 
            }
            rois = Random.shuffle(usable ::: unusable).toList
            validate = (_: RoisSplit.Result) match {
                case tooFew: RoisSplit.TooFewShifting => 
                    tooFew.requestedShifting shouldBe numShifting
                    tooFew.realizedShifting shouldBe NonnegativeInt.unsafe(usable.size)
                    tooFew.realizedAccuracy shouldBe NonnegativeInt(0)
                    tooFew.requestedAccuracy shouldBe numAccuracy
                case res => fail(s"Expected TooFewShifting but got $res")
            }
        } yield (numShifting, numAccuracy, rois, validate)
        
        def genAtLeastMinButLessThanShiftingRequest: Gen[InputsAndValidate] = for {
            numShifting <- // 1 more than absolute min, so that minimum can be hit while not hitting request.
                Gen.choose[Int](AbsoluteMinimumShifting + 1, maxNumRoisSmallTests).map(ShiftingCount.unsafe)
            numAccuracy <- arbitrary[PositiveInt]
            maxUsable = scala.math.max(AbsoluteMinimumShifting, numShifting - 1)
            usable <- genUsableRois(AbsoluteMinimumShifting, maxUsable).map(_.map(_.setUsable))
            unusable <- genUnusableRois(0, maxNumRoisSmallTests - usable.length)
            rois = Random.shuffle(usable ::: unusable).toList
            validate = (_: RoisSplit.Result) match {
                case tooFew: RoisSplit.TooFewAccuracyRescued => 
                    tooFew.requestedShifting shouldBe numShifting
                    tooFew.realizedShifting shouldBe NonnegativeInt.unsafe(usable.size)
                    tooFew.requestedAccuracy shouldBe numAccuracy
                    tooFew.realizedAccuracy shouldBe NonnegativeInt(0)
                case res => fail(s"Expected TooFewAccuracyRescued but got $res")
            }
        } yield (numShifting, numAccuracy, rois, validate)
        
        def genAtLeastShiftingButNotAccuracy: Gen[InputsAndValidate] = for {
            numShifting <- Gen.choose[Int](AbsoluteMinimumShifting, maxNumRoisSmallTests).map(ShiftingCount.unsafe)
            numAccuracy <- arbitrary[PositiveInt]
            maxUsable = scala.math.min(maxNumRoisSmallTests, numShifting + numAccuracy - 1)
            usable <- genUsableRois(numShifting, maxUsable)
            unusable <- genUnusableRois(0, maxNumRoisSmallTests - usable.length)
            rois = Random.shuffle(usable ::: unusable).toList
            validate = (_: RoisSplit.Result) match {
                case tooFew: RoisSplit.TooFewAccuracyHealthy => 
                    tooFew.realizedShifting shouldBe numShifting
                    tooFew.requestedAccuracy shouldBe numAccuracy
                    tooFew.realizedAccuracy shouldBe NonnegativeInt.unsafe(usable.size - numShifting)
                case res => fail(s"Expected TooFewAccuracyHealthy but got $res")
            }
        } yield (numShifting, numAccuracy, rois, validate)
        
        def genEnoughForBoth: Gen[InputsAndValidate] = for {
            numShifting <- Gen.choose(AbsoluteMinimumShifting, maxNumRoisSmallTests - 1).map(ShiftingCount.unsafe)
            numAccuracy <- Gen.choose(1, maxNumRoisSmallTests - numShifting).map(PositiveInt.unsafe)
            usable <- genUsableRois(numShifting + numAccuracy, maxNumRoisSmallTests)
            unusable <- genUnusableRois(0, maxNumRoisSmallTests - usable.length)
            rois = Random.shuffle(usable ::: unusable).toList
            validate = (_: RoisSplit.Result) match {
                case partition: RoisSplit.Partition => 
                    partition.numShifting shouldBe numShifting
                    partition.numAccuracy shouldBe numAccuracy
                case res => fail(s"Expected Partition but got $res")
            }
        } yield (numShifting, numAccuracy, rois, validate)
        
        forAll (
            Gen.oneOf(
                genFewerThanAbsoluteMinimum, 
                genAtLeastMinButLessThanShiftingRequest, 
                genAtLeastShiftingButNotAccuracy, 
                genEnoughForBoth
                ), 
            minSuccessful(10000)
        ) { (numShifting, numAccuracy, rois, validate) => 
            val observation = sampleDetectedRois(numShifting, numAccuracy)(rois)
            validate(observation)
        }
    }

    test("Cases of TooFewHealthyRoisRescued are correct and generate expected (implied) too-few-accuracy-ROIs records.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        val posTimePairs = Random.shuffle(
            (0 to 1).flatMap{ p => (0 to 2).map(p -> _) }
        ).toList.map((p, t) => PositionIndex.unsafe(p) -> ImagingTimepoint.unsafe(t))
        type PosTimeRois = (PosTimePair, List[DetectedRoi])
        def genDetected(ptPairs: List[PosTimePair])(lo: Int, hi: Int): Gen[List[PosTimeRois]] = 
            ptPairs.traverse{ pt => genUsableRois(lo, hi).map(pt -> _) }
        val maxReqShifting = 2 * AbsoluteMinimumShifting
        def genArgs: Gen[(List[PosTimeRois], ShiftingCount, List[PosTimeRois])] = for {
            nTooFewShift <- Gen.choose(1, posTimePairs.length)
            (tooFewPosTimePairs, enoughPosTimePairs) = posTimePairs.splitAt(nTooFewShift)
            tooFew <- genDetected(tooFewPosTimePairs)(AbsoluteMinimumShifting + 1, maxReqShifting - 1)
            numReqShifting <- Gen.choose(tooFew.map(_._2.length).max + 1, maxReqShifting).map(ShiftingCount.unsafe)
            enough <- genDetected(enoughPosTimePairs)(maxReqShifting, 2 * maxReqShifting)
        } yield (tooFew, numReqShifting, enough)
        forAll (genArgs, arbitrary[PositiveInt]) { 
            case ((tooFew, reqShifting, enough), reqAccuracy) =>
                tooFew.map(_._2.length).max < reqShifting shouldBe true
                withTempDirectory{ (tempdir: os.Path) => 
                    /* First, write the input data files. */
                    (tooFew ::: enough).foreach{ case ((p, t), rois) => 
                        writeMinimalInputRoisCsv(rois, tempdir / getInputFilename(p, t))
                    }
                    /* Pretest and workflow execution */
                    val warningsFile = tempdir / "roi_partition_warnings.json"
                    os.exists(warningsFile) shouldBe false
                    workflow(tempdir, reqShifting, reqAccuracy, None)
                    /* Check the effect of having run the workflow */
                    // First, check the existence of the warnings file and parse it.
                    os.exists(warningsFile) shouldBe true
                    given reader: Reader[KeyedProblem] = readWriterForKeyedTooFewProblem
                    val warnings = readJsonFile[List[KeyedProblem]](warningsFile)
                    val obsWarnShifting = warnings.filter(_._2.purpose === Purpose.Shifting)
                    val obsWarnAccuracy = warnings.filter(_._2.purpose === Purpose.Accuracy)
                    // Then, check the too-few-shifting (but rescued) records.
                    obsWarnShifting.length shouldEqual tooFew.length
                    obsWarnShifting.map(_._1).toSet shouldEqual tooFew.map(_._1).toSet
                    obsWarnShifting.map(_._2.numRequested) shouldEqual List.fill(obsWarnShifting.length)(reqShifting)
                    tooFew.map(_.map(_.length)).toMap shouldEqual obsWarnShifting.map(_.map(_.numRealized)).toMap
                    /* Finally, check the too-few-accuracy records. */
                    // Each too-few-shifting (rescued) record generates a too-few-accuracy record.
                    obsWarnAccuracy.length >= tooFew.length shouldBe true
                    val correspondingObsWarnAccuracy = obsWarnShifting.map(_._1).flatMap(obsWarnAccuracy.toMap.get)
                    correspondingObsWarnAccuracy.length shouldEqual tooFew.length
                    // Each too-few-accuracy record has the correct requested and realized counts.
                    correspondingObsWarnAccuracy.map(_.numRequested) shouldEqual List.fill(tooFew.length)(reqAccuracy)
                    correspondingObsWarnAccuracy.map(_.numRealized) shouldEqual List.fill(tooFew.length)(NonnegativeInt(0))
                }
        }
    }

    test("A ROI is never used for more than one purpose.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        val posTimePairs = Random.shuffle(
            (0 to 1).flatMap{ p => (0 to 2).map(p -> _) }
        ).toList.map((p, t) => PositionIndex.unsafe(p) -> ImagingTimepoint.unsafe(t))
        val maxReqShifting = 2 * AbsoluteMinimumShifting
        def genArgs: Gen[(ShiftingCount, PositiveInt, List[(PosTimePair, List[DetectedRoi])])] = for {
            numReqShifting <- Gen.choose(AbsoluteMinimumShifting, maxReqShifting).map(ShiftingCount.unsafe)
            numReqAccuracy <- Gen.choose(1, 100).map(PositiveInt.unsafe)
            numReq = numReqShifting + numReqAccuracy
            rois <- posTimePairs.traverse{ pt => genMixedUsabilityRois(AbsoluteMinimumShifting, 2 * numReq).map(pt -> _) }
        } yield (numReqShifting, numReqAccuracy, rois)
        forAll (genArgs) { (reqShifting, reqAccuracy, ptRoisPairs) => 
            withTempDirectory{ (tempdir: os.Path) => 
                /* First, write the input data files. */
                ptRoisPairs.foreach{ case ((p, t), rois) => 
                    writeMinimalInputRoisCsv(rois, tempdir / getInputFilename(p, t))
                }
                /* Pretest and workflow execution */
                val shiftingFolder = getOutputSubfolder(tempdir)(Purpose.Shifting)
                val accuracyFolder = getOutputSubfolder(tempdir)(Purpose.Accuracy)
                os.exists(shiftingFolder) shouldBe false
                os.exists(accuracyFolder) shouldBe false
                workflow(tempdir, reqShifting, reqAccuracy, None)
                /* Make actual output assertions. */
                val expFilesShifting = ptRoisPairs.map{
                    case ((p, t), _) => (p -> t) -> getOutputFilepath(tempdir)(p, t, Purpose.Shifting)
                }.toMap
                val obsFilesShifting = os.list(shiftingFolder).filter(os.isFile).toSet
                obsFilesShifting shouldEqual expFilesShifting.values.toSet
                val obsFilesAccuracy = ptRoisPairs
                    .map{ case ((p, t), _) => (p -> t) -> getOutputFilepath(tempdir)(p, t, Purpose.Accuracy) }
                    .filter((_, fp) => os.isFile(fp))
                    .toMap
                given rwForShifting: ReadWriter[RoiForShifting] = 
                    SelectedRoi.simpleShiftingRW(ParserConfig.coordinateSequence)
                given rwForAccuracy: ReadWriter[RoiForAccuracy] = 
                    SelectedRoi.simpleAccuracyRW(ParserConfig.coordinateSequence)
                val obsRoisShifting = expFilesShifting.view.mapValues(readJsonFile[List[RoiForShifting]]).toMap
                val obsRoisAccuracy = obsFilesAccuracy.view.mapValues(readJsonFile[List[RoiForAccuracy]]).toMap
                val observedIntersections = obsRoisShifting.toList.flatMap{ 
                    (pt, shiftingRois) => obsRoisAccuracy
                        .get(pt)
                        .map{ accuracyRois => (pt, accuracyRois.toSet & shiftingRois.toSet) }
                }
                // For all intersections that exist (the (FOV, time) pair had both shifting and accuracy ROIs), 
                // the intersection between shifting and accuracy must be nonempty (no ROI re-use).
                observedIntersections.filter(_._2 =!= Set()) shouldEqual List()
            }
        }
    }

    test("ROI counts fewer than absolute minimum results in expected warnings.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        val posTimePairs = Random.shuffle(
            (0 to 1).flatMap{ p => (0 to 2).map(p -> _) }
        ).toList.map((p, t) => PositionIndex.unsafe(p) -> ImagingTimepoint.unsafe(t))
        def genDetected(lo: Int, hi: Int) = 
            (_: List[PosTimePair]).traverse{ pt => genMixedUsabilityRoisEachSize(lo, hi).map(pt -> _) }
        def genArgs = for {
            numTooFew <- Gen.choose(1, posTimePairs.length)
            numReqShifting <- Gen.choose(AbsoluteMinimumShifting, 50).map(ShiftingCount.unsafe)
            numReqAccuracy <- Gen.choose(1, 50).map(PositiveInt.unsafe)
            (posTimePairsForTooFew, posTimePairsForEnough) = Random.shuffle(posTimePairs).splitAt(numTooFew)
            tooFew <- genDetected(1, AbsoluteMinimumShifting - 1)(posTimePairsForTooFew)
            enough <- genDetected(AbsoluteMinimumShifting, 2 * (numReqShifting + numReqAccuracy))(posTimePairsForEnough)
        } yield (numReqShifting, numReqAccuracy, tooFew, enough)
        forAll (genArgs) { (numReqShifting, numReqAccuracy, tooFew, enough) => 
            withTempDirectory{ (tempdir: os.Path) => 
                /* First, write the input data files. */
                (tooFew ::: enough).foreach{ case ((p, t), rois) => 
                    writeMinimalInputRoisCsv(rois, tempdir / getInputFilename(p, t))
                }
                val badWarningsFile = tempdir / "roi_partition_warnings.severe.json"
                os.exists(badWarningsFile) shouldBe false
                /* Make actual output assertions. */
                workflow(tempdir, numReqShifting, numReqAccuracy, None)
                // The expected file should be produced, and it should have the expected data.
                os.isFile(badWarningsFile) shouldBe true
                given readerForKeyedProblem: Reader[KeyedProblem] = readWriterForKeyedTooFewProblem
                val obsWarnSevere = readJsonFile[List[KeyedProblem]](badWarningsFile)
                obsWarnSevere.length shouldEqual tooFew.length
                given ord: Ordering[PosTimePair] = summon[Order[PosTimePair]].toOrdering
                obsWarnSevere.map(_._1).sorted shouldEqual tooFew.map(_._1).sorted // Ignore original ordering.
            }
        }
    }

    test("Warnings file is correct and produced IF AND ONLY IF there is at least one case of too-few-ROIs.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        val posTimePairs = Random.shuffle(
            (0 to 1).flatMap{ p => (0 to 2).map(p -> _) }
        ).toList.map((p, t) => PositionIndex.unsafe(p) -> ImagingTimepoint.unsafe(t))
        def genDetected(lo: Int, hi: Int) = 
            // Make all ROIs usable so that the math about how many will be realized (used) is easier; 
            // in particular, we don't want that math dependent on counting the number of usable vs. unusable ROIs.
            (_: List[PosTimePair]).traverse{ pt => genUsableRois(lo, hi).map(pt -> _) }
        def genArgs = for {
            numTooFewReqShifting <- Gen.oneOf(Gen.const(0), Gen.choose(1, posTimePairs.length))
            numTooFewReqAccuracy <- Gen.oneOf(Gen.const(0), Gen.choose(posTimePairs.length - numTooFewReqShifting, posTimePairs.length))
            // Add one here to the lower bound to leave open--ALWAYS--the possibility of generating too few shifting.
            numReqShifting <- Gen.choose[Int](AbsoluteMinimumShifting + 1, maxNumRoisSmallTests).map(ShiftingCount.unsafe)
            numReqAccuracy <- Gen.choose(1, maxNumRoisSmallTests).map(PositiveInt.unsafe)
            numReq = numReqShifting + numReqAccuracy
            (posTimePairsTooFewShifting, rest) = Random.shuffle(posTimePairs).splitAt(numTooFewReqShifting)
            (posTimePairsTooFewAccuracy, posTimePairsEnough) = rest.splitAt(numTooFewReqAccuracy)
            tooFewShifting <- genDetected(AbsoluteMinimumShifting, numReqShifting - 1)(posTimePairsTooFewShifting)
            tooFewAccuracy <- genDetected(numReqShifting, numReq - 1)(posTimePairsTooFewAccuracy)
            enough <- genDetected(numReq, 2 * numReq)(posTimePairsEnough)
        } yield (numReqShifting, numReqAccuracy, tooFewShifting, tooFewAccuracy, enough)
        forAll (genArgs, minSuccessful(200)) { (numReqShifting, numReqAccuracy, tooFewShifting, tooFewAccuracy, enough) => 
            val tooFew = tooFewShifting ::: tooFewAccuracy
            withTempDirectory{ (tempdir: os.Path) => 
                /* First, write the input data files and do pretest */
                (tooFew ::: enough).foreach{ case ((p, t), rois) => writeMinimalInputRoisCsv(rois, tempdir / getInputFilename(p, t)) }
                val warningsFile = tempdir / "roi_partition_warnings.json"
                os.exists(warningsFile) shouldBe false
                /* Run the workflow */
                workflow(tempdir, numReqShifting, numReqAccuracy, None)
                /* Make actual output assertions. */
                tooFew match {
                    case Nil => os.exists(warningsFile) shouldBe false
                    case _ => 
                        os.exists(warningsFile) shouldBe true
                        given readerForWarnings: Reader[KeyedProblem] = readWriterForKeyedTooFewProblem
                        val warnings = readJsonFile[List[KeyedProblem]](warningsFile)
                        val obsWarnShifting = warnings.filter(_._2.purpose === Purpose.Shifting)
                        val obsWarnAccuracy = warnings.filter(_._2.purpose === Purpose.Accuracy)
                        val expWarnShifting = tooFewShifting.map{ (pt, rois) => 
                            pt -> RoisSplit.Problem.shifting(numReqShifting, NonnegativeInt.unsafe(rois.length))
                        }
                        val expWarnAccuracy = 
                            // Each (FOV, time) pair with too few shifting generates 0 accuracy records, and each (FOV, time) pair 
                            // can also generates a too-few-accuracy record realizing its rois count less shifting request.
                            tooFewShifting.map{ (pt, _) => pt -> RoisSplit.Problem.accuracy(numReqAccuracy, NonnegativeInt(0)) } ::: 
                            tooFewAccuracy.map{ (pt, rois) => pt -> RoisSplit.Problem.accuracy(numReqAccuracy, NonnegativeInt.unsafe(rois.length - numReqShifting)) }
                        /* Check the warning counts. */
                        obsWarnShifting.length shouldEqual tooFewShifting.length
                        obsWarnAccuracy.length shouldEqual (tooFewShifting.length + tooFewAccuracy.length)
                        /* Check the actual problems. */
                        obsWarnShifting.toMap shouldEqual expWarnShifting.toMap
                        def collapseKeyedProblems: List[KeyedProblem] => Map[PosTimePair, NonEmptySet[RoisSplit.Problem]] = {
                            import at.ac.oeaw.imba.gerlich.looptrace.collections.*
                            given orderForPurpose: Order[Purpose] = Order.by{
                                case Purpose.Shifting => 0
                                case Purpose.Accuracy => 1
                            }
                            given orderForProblem: Order[RoisSplit.Problem] = Order.by{
                                problem => (problem.purpose, problem.numRequested, problem.numRealized)
                            }
                            given orderingForProblem: Ordering[RoisSplit.Problem] = orderForProblem.toOrdering
                            _.groupBy(_._1).view.mapValues(_.map(_._2).toSet.toNonEmptySetUnsafe).toMap
                        }
                        collapseKeyedProblems(obsWarnAccuracy) shouldEqual collapseKeyedProblems(expWarnAccuracy)
                }
            }
        }
    }

    test("When shifting request takes up all usable ROIs available, JSON files are still written for accuracy but are empty.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        val posTimePairs = Random.shuffle(
            (0 to 1).flatMap{ p => (0 to 2).map(p -> _) }
        ).toList.map((p, t) => PositionIndex.unsafe(p) -> ImagingTimepoint.unsafe(t))
        def genArgs = for {
            rois <- posTimePairs.traverse{ pt => genUsableRois(AbsoluteMinimumShifting, maxNumRoisSmallTests).map(pt -> _) }
            numShifting <- Gen.choose(rois.map(_._2.length).max, 1000).map(ShiftingCount.unsafe)
            numAccuracy <- Gen.choose(1, 1000).map(PositiveInt.unsafe)
        } yield (numShifting, numAccuracy, rois)
        forAll (genArgs, minSuccessful(500)) { (numReqShifting, numReqAccuracy, allFovTimeRois) =>
            withTempDirectory{ (tempdir: os.Path) => 
                /* First, write the input data files and do pretest. */
                allFovTimeRois.foreach{ case ((p, t), rois) => writeMinimalInputRoisCsv(rois, tempdir / getInputFilename(p, t)) }
                val expAccuracyFiles = allFovTimeRois.map{ case ((p, t), _) => (p -> t) -> getOutputFilepath(tempdir)(p, t, Purpose.Accuracy) }
                expAccuracyFiles.exists((_, f) => os.exists(f)) shouldBe false
                /* Then, execute workflow. */
                workflow(tempdir, numReqShifting, numReqAccuracy, None)
                /* Finally, make assertions. */
                expAccuracyFiles.filterNot((_, f) => os.isFile(f)) shouldEqual List()
                given roiReader: Reader[RoiForAccuracy] = simpleAccuracyRW(ParserConfig.coordinateSequence)
                // Every (FOV, time) pair should have an accuracy ROIs file, but it should be empty.
                expAccuracyFiles.map{ (pt, f) => pt -> readJsonFile[List[RoiForAccuracy]](f) } shouldEqual expAccuracyFiles.map(_._1 -> List())
            }
        }
    }

    test("No unusable ROI is ever used, and ROI indices and coordiantes are preserved during partition.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        def genSinglePosTimeRois = {
            given arbPt: Arbitrary[Point3D] = {
                given arbX: Arbitrary[XCoordinate] = Gen.choose(-3e3, 3e3).map(XCoordinate.apply).toArbitrary
                given arbY: Arbitrary[YCoordinate] = Gen.choose(-3e3, 3e3).map(YCoordinate.apply).toArbitrary
                given arbZ: Arbitrary[ZCoordinate] = Gen.choose(-3e3, 3e3).map(ZCoordinate.apply).toArbitrary
                Gen.zip(arbitrary[XCoordinate], arbitrary[YCoordinate], arbitrary[ZCoordinate]).map(Point3D.apply.tupled).toArbitrary
            }
            for {
                usable <- genUsableRois(AbsoluteMinimumShifting, 50)
                unusable <- genUnusableRois(1, 50)
            } yield Random.shuffle(usable ::: unusable).toList
        }
        val posTimePairs = Random.shuffle(
            (0 to 1).flatMap{ p => (0 to 2).map(p -> _) }
        ).toList.map((p, t) => PositionIndex.unsafe(p) -> ImagingTimepoint.unsafe(t))
        def genArgs = for {
            numShifting <- Gen.choose(AbsoluteMinimumShifting, 50).map(ShiftingCount.unsafe)
            numAccuracy <- Gen.choose(1, 50).map(PositiveInt.unsafe)
            rois <- posTimePairs.traverse{ pt => genSinglePosTimeRois.map(pt -> _) }
        } yield (numShifting, numAccuracy, rois)
        val simplifyRoi = (roi: RoiLike) => roi.index -> roi.centroid
        forAll (genArgs, minSuccessful(200)) { (numShifting, numAccuracy, allFovTimeRois) => 
            withTempDirectory{ (tempdir: os.Path) =>
                /* First, write the input data files and do pretest. */
                allFovTimeRois.foreach{ case ((p, t), rois) => writeMinimalInputRoisCsv(rois, tempdir / getInputFilename(p, t)) }
                val expAllOutFiles = allFovTimeRois.map{ case ((p, t), _) => 
                    (p -> t) -> List(getOutputFilepath(tempdir)(p, t, Purpose.Shifting), getOutputFilepath(tempdir)(p, t, Purpose.Accuracy))
                }
                expAllOutFiles.flatMap(_._2).filter(os.isFile) shouldEqual List()
                /* Run the workflow and find the output files. */
                workflow(tempdir, numShifting, numAccuracy, None)
                val obsFilesShifting = os.list(getOutputSubfolder(tempdir)(Purpose.Shifting)).filter(os.isFile).toSet
                val obsFilesAccuracy = os.list(getOutputSubfolder(tempdir)(Purpose.Accuracy)).filter(os.isFile).toSet
                
                val obsFilesAll = obsFilesShifting | obsFilesAccuracy
                expAllOutFiles.flatMap(_._2).toSet shouldEqual obsFilesAll
                
                given readerForShifting: Reader[RoiForShifting] = simpleShiftingRW(ParserConfig.coordinateSequence)
                given readerForAccuracy: Reader[RoiForAccuracy] = simpleAccuracyRW(ParserConfig.coordinateSequence)
                val obsRoisShifting = allFovTimeRois.map{ 
                    case ((p, t), _) => 
                        val raw = readJsonFile[List[RoiForShifting]](getOutputFilepath(tempdir)(p, t, Purpose.Shifting))
                        (p -> t) -> raw.map(simplifyRoi).toSet
                }.toMap
                val obsRoisAccuracy = allFovTimeRois.map{
                    case ((p, t), _) => 
                        val raw = readJsonFile[List[RoiForAccuracy]](getOutputFilepath(tempdir)(p, t, Purpose.Accuracy))
                        (p -> t) -> raw.map(simplifyRoi).toSet
                }.toMap
                val obsRoisAll = obsRoisShifting |+| obsRoisAccuracy
                val (obsRoisUsable, obsRoisUnusable) = {
                    val (usable, unusable) = allFovTimeRois.map{ (pt, rois) => 
                        val (yes, no) = rois.partition(_.isUsable)
                        (pt -> yes.map(simplifyRoi).toSet, pt -> no.map(simplifyRoi).toSet)
                    }.unzip
                    (usable.toMap, unusable.toMap)
                }

                /* Each (FOV, time) */
                obsRoisAll.filter(_._2.isEmpty) shouldEqual Map()
                obsRoisAll.map((pt, rois) => pt -> (rois & obsRoisUnusable(pt)) ) shouldEqual obsRoisAll.map((pt, _) => pt -> Set())
                obsRoisAll.map((pt, rois) => pt -> (rois -- obsRoisUsable(pt)) ) shouldEqual obsRoisAll.map((pt, _) => pt -> Set())
            }
        }
    }

    /* *******************************************************************************
     * Ancillary types and functions
     * *******************************************************************************
     */
    type NNPair = (NonnegativeInt, NonnegativeInt)

    /** Minimal detected bead ROIs field consumed by the partitioning program under test */
    val ColumnNamesToParse = List(ParserConfig.xCol.get, ParserConfig.yCol.get, ParserConfig.zCol.get, ParserConfig.qcCol)

    /**
     * Generate collection of detected ROIs in which usability is mixed, for tests where percentage/ratio should be irrelevant.
     * 
     * Here, note that the _total_ size of the generated collection will be in [lo,  hi].
     * 
     * @param lo The minimum number of ROIs in the output collection
     * @param hi The maximum number of ROIs in the output collection
     * @return A generator of a collection of detected ROIs
     */
    def genMixedUsabilityRois(lo: Int, hi: Int)(using Arbitrary[RoiIndex], Arbitrary[Point3D]) = for {
        usable <- genUsableRois(lo, hi)
        unusable <- genUnusableRois(math.max(0, lo - usable.length), hi - usable.length)
    } yield Random.shuffle(usable ::: unusable).toList

    /**
     * Generate collection of detected ROIs in which usability is mixed, for tests where percentage/ratio should be irrelevant.
     * 
     * Here, note that the size of _each_ subcollection (usable or unusable) will be in [lo,  hi].
     * Therefore, the _total_ collection size will be in [2 * lo, 2 * hi].
     * 
     * @param lo The minimum number of ROIs in _each_ of the subcollections (usable and unusable)
     * @param hi The maximum number of ROIs in _each_ of the subcollections (usable and unusable)
     * @return A generator of a collection of detected ROIs
     */
    def genMixedUsabilityRoisEachSize(lo: Int, hi: Int)(using Arbitrary[RoiIndex], Arbitrary[Point3D]) = for {
        usable <- genUsableRois(lo, hi)
        unusable <- genUnusableRois(lo, hi)
    } yield Random.shuffle(usable ::: unusable).toList

    /** Generate {@code [lo, hi]} detected ROIs with nonempty fail code. */
    def genUnusableRois(lo: Int, hi: Int)(using Arbitrary[RoiIndex], Arbitrary[Point3D]) = 
        Gen.choose(lo, hi).flatMap(Gen.listOfN(_, genUnusableDetectedRoi))

    /** Generate {@code [lo, hi]} detected ROIs with empty fail code. */
    def genUsableRois(lo: Int, hi: Int)(using Arbitrary[RoiIndex], Arbitrary[Point3D]) = 
        genUnusableRois(lo, hi).map(_.map(_.setUsable))

    /** Generate a single {@code DetectedRoi} with nonempty fail code. */
    def genUnusableDetectedRoi(using Arbitrary[Point3D], Arbitrary[RoiIndex]): Gen[DetectedRoi] = for {
        i <- arbitrary[RoiIndex]
        pt <- arbitrary[Point3D]
        failCode <- Gen.choose(1, 5).flatMap(Gen.listOfN(_, Gen.alphaChar).map(_.mkString("")))
    } yield DetectedRoi(i, pt, RoiFailCode(failCode))

    /** Syntax additions on a detected ROI to set its usability flag */
    extension (roi: DetectedRoi)
        def setUsable: DetectedRoi = roi.copy(failCode = RoiFailCode.success)

    /** Generate a pair of pairs of nonnegative integers such that the first pair isn't the same as the second. */
    def genDistinctNonnegativePairs: Gen[(PosTimePair, PosTimePair)] = 
        Gen.zip(arbitrary[(NonnegativeInt, NonnegativeInt)], arbitrary[(NonnegativeInt, NonnegativeInt)])
            .suchThat{ case (p1, p2) => p1 =!= p2 }
            .map { case ((p1, f1), (p2, f2)) => (PositionIndex(p1) -> ImagingTimepoint(f1), PositionIndex(p2) -> ImagingTimepoint(f2)) }
    
    /** Infer detected bead ROIs filename for particular field of view (@code pos) and timepoint ({@code time}). */
    def getInputFilename(pos: PositionIndex, time: ImagingTimepoint): String = s"bead_rois__${pos.get}_${time.get}.csv"
    
    /** Limit the number of ROIs generated to keep test cases (relatively) small even without shrinking. */
    def maxNumRoisSmallTests: ShiftingCount = ShiftingCount.unsafe(2 * AbsoluteMinimumShifting)

    /** Write the ROIs to file, with minimal data required to parse the fields consumed by the partition program under test here. */
    def writeMinimalInputRoisCsv(rois: List[DetectedRoi], f: os.Path): Unit = {
        val (header, getPointFields) = (
            Array("", ParserConfig.xCol.get, ParserConfig.yCol.get, ParserConfig.zCol.get, ParserConfig.qcCol),
            (p: Point3D) => Array(p.x.get, p.y.get, p.z.get).map(_.toString)
        )
        val records = rois.map{ roi => roi.index.get.toString +: getPointFields(roi.centroid) :+ roi.failCode.get }
        os.write(f, (header +: records).map(_.mkString(",") ++ "\n"))
    }

end TestPartitionIndexedDriftCorrectionBeadRois
