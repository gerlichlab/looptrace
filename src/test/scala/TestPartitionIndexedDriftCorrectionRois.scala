package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ Random, Try }

import cats.data.{ NonEmptyList as NEL }
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*

import at.ac.oeaw.imba.gerlich.looptrace.PartitionIndexedDriftCorrectionRois.{
    AbsoluteMinimumShifting,
    BadRecord,
    BeadRoisPrefix, 
    ColumnName,
    InitFile,
    ParserConfig,
    PosFramePair,
    Purpose,
    RoisFileParseFailedRecords,
    RoisFileParseFailedSetup,
    RoisSplit,
    ShiftingCount,
    XColumn, 
    YColumn, 
    ZColumn, 
    discoverInputs, 
    getOutputFilepath,
    getOutputSubfolder,
    readRoisFile,
    sampleDetectedRois, 
    workflow, 
}
import at.ac.oeaw.imba.gerlich.looptrace.PathHelpers.listPath
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.readJsonFile
import at.ac.oeaw.imba.gerlich.looptrace.space.{ CoordinateSequence, Point3D, XCoordinate, YCoordinate, ZCoordinate }

/** Tests for the partitioning of regions of interest (ROIs) for drift correction */
class TestPartitionIndexedDriftCorrectionRois extends AnyFunSuite, ScalacheckSuite, should.Matchers, PartitionRoisSuite:
    import SelectedRoi.*
    
    test("ShiftingCount appropriately constrains the domain.") { pending }

    test("Cannot request to sample fewer than minimum number of ROIs, won't compile.") { pending }

    test("Any case of too few shifting ROIs is also a case of too few accuracy ROIs.") { pending }

    test("Types properly restrict compilation.") { pending }

    test("Cannot mixup shifting and accuracy count values") { pending }

    test("Cannot mixup requested and realized values") { pending }

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
                readRoisFile(roisFile) shouldEqual Left(NEL.one(s"No lines in file! $roisFile"))
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
                    val expErrorMessages = NEL(s"Cannot infer delimiter for file! $roisFile", extras)
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
                NEL.one(BadRecord(NonnegativeInt(1), delimiter.split("104,105,10.6,1919.8,1137.4,12858.0,5.0,,"), NEL.one("Header has 8 fields but record has 9")))
            ),
            (
                headLine :: List(
                    "101,102,11.96875,1857.9375,1076.25,26799.0,32.0,", 
                    "104,105,10.6,1919.8,1137.4,12858.0,5.0"
                    ), 
                NEL.one(BadRecord(NonnegativeInt(1), delimiter.split("104,105,10.6,1919.8,1137.4,12858.0,5.0"), NEL.one("Header has 8 fields but record has 7")))
            ), 
            (
                headLine :: List(
                    "101,102,11.96875,1857.9375,1076.25,26799.0,32.0", 
                    "104,105,10.6,1919.8,1137.4,12858.0,5.0,i", 
                    "109,107,11.96875,1857.9375,1076.25,26799.0"
                    ), 
                NEL(
                    BadRecord(NonnegativeInt(0), delimiter.split("101,102,11.96875,1857.9375,1076.25,26799.0,32.0"), NEL.one("Header has 8 fields but record has 7")), 
                    List(BadRecord(NonnegativeInt(2), delimiter.split("109,107,11.96875,1857.9375,1076.25,26799.0"), NEL.one("Header has 8 fields but record has 6")))
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

    test("Header-only file parses but yields empty record collection.") {
        forAll(minSuccessful(1000)) { (delimiter: Delimiter) => 
            val headLine = delimiter.join(ColumnNamesToParse.toArray) ++ "\n"
            withTempFile(headLine, delimiter){ (roisFile: os.Path) => 
                readRoisFile(roisFile) shouldEqual Right(List())
            }
        } 
    }

    test("Input discovery works as expected for folder with no other contents.") {
        forAll (genDistinctNonnegativePairs) { case (pf1, pf2) => {
            withTempDirectory{ (p: os.Path) => 
                val expected = Set(pf1, pf2).map(pf => pf -> (p / getInputFilename(pf._1, pf._2)))

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
        def setup(root: os.Path, pf1: PosFramePair, pf2: PosFramePair, fc: FolderChoice): (os.Path, Set[InitFile]) = {
            import FolderChoice.*
            val subGood = root / "sub1"
            val subBad = root / "sub2"
            val (pos, frame) = pf1
            val wrongPrefixFile = subGood / getInputFilename(pos, frame).replaceAll(BeadRoisPrefix, "BadPrefix")
            val wrongSubfolderFile = subBad / getInputFilename(pos, frame)
            val missingPrefixFile = subGood / getInputFilename(pos, frame).replaceAll(BeadRoisPrefix, "")
            val wrongFilenameStructureFile1 = 
                subGood / getInputFilename(pos, frame).replaceAll(s"${pos.get}_${frame.get}", s"${pos.get}.${frame.get}")
            val wrongFilenameStructureFile2 = 
                subGood / getInputFilename(pos, frame).replaceAll(s"${pos.get}_${frame.get}", s"${pos.get}_${frame.get}_0")
            val goodFile1 = subGood / getInputFilename(pos, frame)
            val goodFile2 = subGood / getInputFilename(pf2._1, pf2._2)
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
                case GoodSubfolder => subGood -> Set(((pos, frame), goodFile1), ((pf2._1, pf2._2), goodFile2))
                case BadSubfolder => subBad -> Set(((pos, frame), wrongSubfolderFile))
            }
        }
        forAll (Gen.zip(genDistinctNonnegativePairs, Gen.oneOf(FolderChoice.values.toList))) { 
            case ((pf1, pf2), folderChoice) => 
                withTempDirectory({ (root: os.Path) => 
                    val (inputPath, expectedOutput) = setup(root, pf1, pf2, folderChoice)
                    
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
        extension (roi: DetectedRoi)
            def setUsable: DetectedRoi = roi.copy(isUsable = true)
            def setUnusable: DetectedRoi = roi.copy(isUsable = false)
        
        def genFewerThanAbsoluteMinimum: Gen[InputsAndValidate] = for {
            numShifting <- Gen.choose[Int](AbsoluteMinimumShifting, maxNumRoisSmallTests).map(ShiftingCount.unsafe)
            numAccuracy <- arbitrary[PositiveInt]
            (usable, unusable) <- (for {
                goods <- genRois(0, AbsoluteMinimumShifting - 1).map(_.map(_.setUsable))
                bads <- genRois(0, maxNumRoisSmallTests - goods.length).map(_.map(_.setUnusable))
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
        
        def genRois(lo: Int, hi: Int) = Gen.choose(lo, hi).flatMap(Gen.listOfN(_, arbitrary[DetectedRoi]))
        
        def genAtLeastMinButLessThanShiftingRequest: Gen[InputsAndValidate] = for {
            numShifting <- // 1 more than absolute min, so that minimum can be hit while not hitting request.
                Gen.choose[Int](AbsoluteMinimumShifting + 1, maxNumRoisSmallTests).map(ShiftingCount.unsafe)
            numAccuracy <- arbitrary[PositiveInt]
            maxUsable = scala.math.max(AbsoluteMinimumShifting, numShifting - 1)
            usable <- genRois(AbsoluteMinimumShifting, maxUsable).map(_.map(_.setUsable))
            unusable <- genRois(0, maxNumRoisSmallTests - usable.length).map(_.map(_.setUnusable))
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
            usable <- genRois(numShifting, maxUsable).map(_.map(_.setUsable))
            unusable <- genRois(0, maxNumRoisSmallTests - usable.length).map(_.map(_.setUnusable))
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
            usable <- genRois(numShifting + numAccuracy, maxNumRoisSmallTests).map(_.map(_.setUsable))
            unusable <- genRois(0, maxNumRoisSmallTests - usable.length).map(_.map(_.setUnusable))
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

    // test("An ROI is never used for more than one purpose.") {
    //     val maxRoisCount = PositiveInt(1000)
    //     def genGoodInput: Gen[(PositiveInt, PositiveInt, Iterable[DetectedRoi])] = for {
    //         numUsable <- Gen.choose(2, maxRoisCount - 1)
    //         usable <- Gen.listOfN(numUsable, genDetectedRoiFixedUse(true))
    //         numUnusable <- Gen.choose(1, maxRoisCount - numUsable)
    //         unusable <- Gen.listOfN(numUnusable, genDetectedRoiFixedUse(false))
    //         numShifting <- Gen.choose(1, numUsable - 1).map(PositiveInt.unsafe)
    //         numAccuracy <- Gen.choose(1, maxRoisCount).map(PositiveInt.unsafe)
    //     } yield (numShifting, numAccuracy, Random.shuffle(usable ++ unusable))
        
    //     def simplifyRoi(roi: RoiForShifting | RoiForAccuracy): (RoiIndex, Point3D) = roi.index -> roi.centroid

    //     forAll (genGoodInput, minSuccessful(1000)) { case (numShifting, numAccuracy, rois) => 
    //         sampleDetectedRois(numShifting, numAccuracy)(rois) match {
    //             case result: RoisSplit.Failure => fail(s"Expected successful partition but got failure: $result")
    //             case result: RoisSplit.HasPartition => 
    //                 val part = result.partition
    //                 part.shifting.length shouldEqual part.shifting.toNes.size // no duplicates within shifting
    //                 part.accuracy.length shouldEqual part.accuracy.toSet.size // no duplicates within accuracy
    //                 (part.shifting.map(simplifyRoi).toSet & part.accuracy.map(simplifyRoi).toSet) shouldEqual Set()
    //         }
    //     }
    // }

    // test("Integration: toggle for tolerance of insufficient shifting ROIs works.") {
    //     /**
    //      * In this test, we generate cases in which it's possible that either one or both datasets
    //      * have sufficient ROI counts for the randomly generated shifting and accuracy ROI counts, 
    //      * or the one or both of the datasets have insufficient ROIs for the shifting and/or 
    //      * accuracy requests. The tolerance for insufficient shifting ROIs is also randomised, 
    //      * and expected output files present, and expected contents, are accordingly adjusted.
    //     */
    //     import SmallDataSet.*

    //     implicit def noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

    //     type PF = PosFramePair
    //     val N1 = input1.points.length
    //     val N2 = input2.points.length

    //     def genSampleSizes: Gen[(PositiveInt, PositiveInt)] = for {
    //         numShifting <- Gen.choose(1, scala.math.min(N1, N2) - 1).map(PositiveInt.unsafe)
    //         numAccuracy <- Gen.choose(1, scala.math.max(N1, N2) + 1).map(PositiveInt.unsafe)
    //     } yield (numShifting, numAccuracy)
        
    //     def gen5PF = arbitrary[(PF, PF, PF, PF, PF)].suchThat{ case (a, b, c, d, e) => Set(a, b, c, d, e).size === 5 }
        
    //     given inputArb: Arbitrary[InputBundle] = Arbitrary{ Gen.oneOf(input1, input2) }
    //     def genInputsWithPF = Gen.zip(arbitrary[(InputBundle, InputBundle, InputBundle, InputBundle, InputBundle)], gen5PF) map {
    //         case ((in1, in2, in3, in4, in5), (pf1, pf2, pf3, pf4, pf5)) => ((pf1, in1), (pf2, in2), (pf3, in3), (pf4, in4), (pf5, in5))
    //     }

    //     def genInputsAndExpectation = for {
    //         (numShifting, numAccuracy) <- genSampleSizes
    //         (a@(pf1, in1), b@(pf2, in2), c@(pf3, in3), d@(pf4, in4), e@(pf5, in5)) <- genInputsWithPF
    //         usedFrames = Set(pf1._2, pf2._2, pf3._2, pf4._2, pf5._2)
    //         useOneAsRef <- Gen.oneOf(false, true)
    //         refFrame <- (
    //             if useOneAsRef 
    //             then Gen.oneOf(usedFrames).map(_.some)
    //             else Gen.option(arbitrary[FrameIndex]).suchThat(_.fold(true)(i => !usedFrames.contains(i)))
    //             )
    //         (fatal, nonfatal) = List(a, b, c, d, e).foldRight(List.empty[(PF, InputBundle, TooFewShifting)], List.empty[(PF, InputBundle, TooFewAccuracy)]) { 
    //             case ((pf, in), (worse, bads)) => 
    //                 if in.numUsable < numShifting then ((pf, in, TooFewShifting(numShifting, in.numUsable)) :: worse, bads)
    //                 else if in.numUsable < numShifting + numAccuracy then 
    //                     // dummy null partition here, since it should never be accessed (only care about the requested and realised counts)
    //                     val err = TooFewAccuracy(null, numAccuracy, NonnegativeInt.unsafe(in.numUsable - numShifting))
    //                     (worse, (pf, in, err) :: bads)
    //                 else (worse, bads)
    //             }
    //         (expError, expSevere) = (fatal, refFrame) match {
    //             case (Nil, _) => (None, None)
    //             case (_, None) => (Exception(s"${fatal.size} (position, frame) pairs with problems.\n${fatal.map(t => t._1 -> t._3)}").some, None)
    //             case (_, Some(rf)) => fatal.partition(_._1._2 === rf) match {
    //                 case (Nil, tolerated) => (None, tolerated.map(t => t._1 -> t._3).some)
    //                 case (untolerated, _) => (Exception(s"${untolerated.size} (position, frame) pairs with problems.\n${untolerated.map(t => t._1 -> t._3)}").some, None)
    //             }
    //         }
    //         expWarn = (expError.isEmpty && nonfatal.nonEmpty).option{ nonfatal.map(t => t._1 -> t._3) }
    //     } yield (List(a, b, c, d, e), numShifting, numAccuracy, refFrame, (expError, expSevere, expWarn))

    //     forAll (genInputsAndExpectation, minSuccessful(1000)) { 
    //         case (inputsWithPF, numShifting, numAccuracy, refFrame, (expError, expSevere, expWarn)) => 
    //             withTempDirectory{ (tempdir: os.Path) =>
    //                 inputsWithPF.foreach(writeBundle(tempdir).tupled) // Prep the data.
    //                 expError match {
    //                     case Some(exc) => assertThrows[Exception]{ workflow(tempdir, numShifting, numAccuracy, refFrame, None) }
    //                     case None => 
    //                         workflow(tempdir, numShifting, numAccuracy, refFrame, None)
    //                         assertTooFewRoisFileContents(tempdir / "roi_partition_warnings.severe.json", expSevere)
    //                         assertTooFewRoisFileContents(tempdir / "roi_partition_warnings.json", expWarn)
    //                 }
    //             }
    //     }
    // }

    // test("Integration: shifting <= #(usable ROIs) < shifting + accuracy ==> warnings file correctly produced; #116") {
    //     import SmallDataSet.*
    //     given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

    //     def genInputs = for {
    //         numShifting <- Gen.choose(1, maxRequestNum - 1).map(PositiveInt.unsafe)
    //         numAccuracy <- Gen.choose(maxRequestNum - numShifting + 1, TooHighRoisNum).map(PositiveInt.unsafe)
    //         maybeSubfolderName <- Gen.option(Gen.const("temporary_subfolder"))
    //         pf1 <- arbitrary[PosFramePair]
    //         pf2 <- arbitrary[PosFramePair].suchThat(_ =!= pf1)
    //     } yield (numShifting, numAccuracy, maybeSubfolderName, pf1, pf2)

    //     forAll (genInputs) { case (numShifting, numAccuracy, maybeSubfolderName, pf1, pf2) =>
    //         withTempDirectory{ (tempdir: os.Path) =>
    //             /* Setup the inputs. */
    //             List(pf1 -> input1, pf2 -> input2).foreach(writeBundle(tempdir).tupled)
    //             /* Check that the workflow creates the expected warnings file. */
    //             val warningsFile = tempdir / "roi_partition_warnings.json"
    //             os.exists(warningsFile) shouldBe false
    //             workflow(inputRoot = tempdir, numShifting = numShifting, numAccuracy = numAccuracy)
    //             os.isFile(warningsFile) shouldBe true
    //         }
    //     }
    // }

    // test("Integration: golden path's overall behavioral properties are correct.") {
    //     import SmallDataSet.*
    //     given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

    //     /** Generate shifting and accuracy counts that should yield no warnings and no errors. */
    //     def genInputs = for {
    //         coordseq <- arbitrary[CoordinateSequence]
    //         numShifting <- Gen.choose(1, maxRequestNum - 1).map(PositiveInt.unsafe)
    //         numAccuracy <- Gen.choose(1, maxRequestNum - numShifting).map(PositiveInt.unsafe)
    //         maybeSubfolderName <- Gen.option(Gen.const("temporary_subfolder"))
    //         pf1 <- arbitrary[PosFramePair]
    //         pf2 <- arbitrary[PosFramePair].suchThat(_ =!= pf1)
    //     } yield (coordseq, numShifting, numAccuracy, maybeSubfolderName, pf1, pf2)
        
    //     forAll (genInputs) {
    //         case (coordseq, numShifting, numAccuracy, maybeSubfolderName, pf1, pf2) => 
    //             withTempDirectory{ (tempdir: os.Path) =>
    //                 /* Setup the inputs. */
    //                 val roisFolder = maybeSubfolderName.fold(tempdir)(tempdir / _)
    //                 os.makeDir.all(roisFolder)
    //                 val roiFileParts = List(pf1 -> input1, pf2 -> input2).map { 
    //                     case (pf, inputBundle) => 
    //                         val fp = roisFolder / getInputFilename.tupled(pf)
    //                         os.write(fp, inputBundle.lines.mkString("\n"))
    //                         pf -> inputBundle.partition
    //                     }.toMap
                    
    //                 workflow(inputRoot = roisFolder, numShifting = numShifting, numAccuracy = numAccuracy)
    //                 val shiftingOutfolder = getOutputSubfolder(roisFolder)(Purpose.Shifting)
    //                 val accuracyOutfolder = getOutputSubfolder(roisFolder)(Purpose.Accuracy)
    //                 List(shiftingOutfolder, accuracyOutfolder).forall(os.isDir) shouldBe true
                    
    //                 val posFramePairs = List(pf1, pf2)
    //                 val expShiftingOutfiles = posFramePairs.map{ case (p, f) => (p, f) -> getOutputFilepath(roisFolder)(p, f, Purpose.Shifting) }
    //                 val expAccuracyOutfiles = posFramePairs.map{ case (p, f) => (p, f) -> getOutputFilepath(roisFolder)(p, f, Purpose.Accuracy) }
    //                 given shiftingRoiRW: ReadWriter[RoiForShifting] = SelectedRoi.simpleShiftingRW(coordseq)
    //                 given accuracyRoiRW: ReadWriter[RoiForAccuracy] = SelectedRoi.simpleAccuracyRW(coordseq)
    //                 val obsShiftings = expShiftingOutfiles.map { case (pf, fp) => (pf, readJsonFile[List[RoiForShifting]](fp)) }.toMap
    //                 val obsAccuracies = expAccuracyOutfiles.map { case (pf, fp) => pf -> readJsonFile[List[RoiForAccuracy]](fp) }.toMap
                    
    //                 val shiftingOutFiles = os.list(shiftingOutfolder)
    //                 val accuracyOutFiles = os.list(accuracyOutfolder)
    //                 shiftingOutFiles.length shouldBe posFramePairs.length
    //                 accuracyOutFiles.length shouldBe posFramePairs.length
    //                 shiftingOutFiles.toSet shouldEqual expShiftingOutfiles.map(_._2).toSet
    //                 accuracyOutFiles.toSet shouldEqual expAccuracyOutfiles.map(_._2).toSet
    //                 obsShiftings.keySet shouldEqual obsAccuracies.keySet
    //                 obsShiftings.forall(_._2.length === numShifting) shouldBe true
    //                 obsAccuracies.forall(_._2.length === numAccuracy) shouldBe true
                    
    //                 val byPosFrame = obsShiftings.map{ case (pf, shifting) => pf -> (shifting, obsAccuracies(pf)) }
                    
    //                 // Set of ROIs for shifting must have no overlap with ROIs for accuracy.
    //                 byPosFrame.values.map{ case (shifting, accuracy) => 
    //                     (shifting.map(_.index).toSet & accuracy.map(_.index).toSet) 
    //                 } shouldEqual List.fill(posFramePairs.length)(Set()) // no intersection
                    
    //                 // Each selected ROI (shifting and accuracy) should have been in the usable pool, not unusable.
    //                 byPosFrame.toList.map{ case (pf, (shifting, accuracy)) => 
    //                     val delIdx = shifting.map(_.index).toSet
    //                     val accIdx = accuracy.map(_.index).toSet
    //                     val part = roiFileParts(pf)
    //                     (delIdx.forall(part.usable.contains), !delIdx.exists(part.unusable.contains), accIdx.forall(part.usable.contains), !accIdx.exists(part.unusable.contains))
    //                 } shouldEqual List.fill(posFramePairs.length)((true, true, true, true))
                    
    //                 // The coordinates of the ROI centroids must be preserved during the selection/partitioning.
    //                 byPosFrame.toList.map{ case (pf, (shifting, accuracy)) => 
    //                     val getPoint = roiFileParts(pf).getPointSafe
    //                     shifting.filterNot(roi => roi.centroid.some === getPoint(roi.index)) -> accuracy.filterNot(roi => roi.centroid.some === getPoint(roi.index))
    //                 } shouldEqual List.fill(posFramePairs.length)(List() -> List())

    //                 // There should be no warnings, anywhere.
    //                 PathHelpers.listPath(tempdir).filter(_.last === "roi_partition_warnings.json").isEmpty shouldBe true
    //             }
    //     }
    // }

    /** Bundles of test data to use for the integration-like tests here, doing more actual file I/O */
    object SmallDataSet:
        val TooHighRoisNum = 10000

        final case class InputBundle(lines: List[String], partition: Partition):
            final def points = partition.points
            final def numUsable = NonnegativeInt.unsafe(partition.numUsable)
        end InputBundle
        
        final case class Partition(points: List[Point3D], usable: Set[RoiIndex]):
            require(points.nonEmpty, "No points for partition!")
            require(usable.size > 1, "Need at least 2 usable ROIs!")
            require(usable.forall(_.get < points.length), "Illegal usability indices!") // Ensure each index corresponds to a point.
            final def getPointSafe = (i: RoiIndex) => Try{ points(i.get) }.toOption
            final def numUsable = usable.size
            final def unusable: Set[RoiIndex] = (0 to points.size).map(RoiIndex.unsafe).toSet -- usable
        end Partition

        def input1 = InputBundle(lines1, indexPartition1)
        private def points1 = List(
            (11.96875, 1857.9375, 1076.25),
            (10.6, 1919.8, 1137.4),
            (11.88, 1939.52, 287.36),
            (11.5, 1942.0, 1740.625),
            (11.35, 2031.0, 863.15),
            (12.4, 6.4, 1151.5), 
            (12.1, 8.1, 1709.5)
            ).map(buildPoint.tupled)
        private def indexPartition1 = Partition(points1, usable = Set(0, 2, 3, 4).map(RoiIndex.unsafe))
        private def lines1 = """,label,centroid-0,centroid-1,centroid-2,max_intensity,area,fail_code
            101,102,11.96875,1857.9375,1076.25,26799.0,32.0,
            104,105,10.6,1919.8,1137.4,12858.0,5.0,i
            109,110,11.88,1939.52,287.36,21065.0,25.0,
            110,111,11.5,1942.0,1740.625,21344.0,32.0,
            115,116,11.35,2031.0,863.15,19610.0,20.0,
            116,117,12.4,6.4,1151.5,16028.0,10.0,y
            117,118,12.1,8.1,1709.5,14943.0,10.0,i
            """.split("\n").map(_.trim).toList
        
        def input2 = InputBundle(lines2, indexPartition2)
        private def points2 = List(
                (9.6875, 888.375, 1132.03125),
                (10.16, 1390.94, 1386.96),
                (9.3125, 1567.5, 87.40625),
                (9.166666666666666, 1576.75, 18.0),
                (9.0, 1725.4, 1886.8),
                (9.0, 1745.6, 1926.1),
                (8.875, 1851.25, 1779.5),
                (9.708333333333334, 1831.625, 1328.0),
                ).map(buildPoint.tupled)
        private def indexPartition2 = Partition(points2, usable = Set(0, 1, 2, 3, 5, 7).map(RoiIndex.unsafe))
        private def lines2 = """,label,centroid-0,centroid-1,centroid-2,max_intensity,area,fail_code
            3,4,9.6875,888.375,1132.03125,25723.0,32.0,
            20,21,10.16,1390.94,1386.96,33209.0,50.0,
            34,35,9.3125,1567.5,87.40625,23076.0,32.0,
            36,37,9.166666666666666,1576.75,18.0,16045.0,12.0,
            41,42,9.0,1725.4,1886.8,12887.0,5.0,i
            43,44,9.0,1745.6,1926.1,15246.0,10.0,
            44,47,8.875,1851.25,1779.5,14196.0,8.0,i
            46,45,9.708333333333334,1831.625,1328.0,22047.0,24.0,
            """.split("\n").map(_.trim).toList
        
        def delimiter = Delimiter.CommaSeparator
        def maxRequestNum = scala.math.min(indexPartition1.usable.size, indexPartition2.usable.size)
        private def buildPoint(x: Double, y: Double, z: Double) = Point3D(XCoordinate(x), YCoordinate(y), ZCoordinate(z))
    end SmallDataSet

    /* *******************************************************************************
     * Ancillary types and functions
     * *******************************************************************************
     */
    type NNPair = (NonnegativeInt, NonnegativeInt)

    val ColumnNamesToParse = List(ParserConfig.xCol.get, ParserConfig.yCol.get, ParserConfig.zCol.get, ParserConfig.qcCol)
    
    // def assertTooFewRoisFileContents[A](filepath: os.Path, expected: Option[List[(PosFramePair, A)]]) = {
    //     expected match {
    //         case None => os.exists(filepath) shouldBe false
    //         case Some(pfTooFewPairs) => 
    //             os.isFile(filepath) shouldBe true
    //             val obs = readJsonFile[List[(PosFramePair, TooFewRois)]](filepath)
    //             val exp = pfTooFewPairs.map{ case (pf, tooFew) => pf -> tooFew.problem }
    //             obs.length shouldEqual exp.length
    //             obs.toSet shouldEqual exp.toSet
    //     }
    // }

    def genDistinctNonnegativePairs: Gen[(PosFramePair, PosFramePair)] = 
        Gen.zip(genNonnegativePair, genNonnegativePair)
            .suchThat{ case (p1, p2) => p1 =!= p2 }
            .map { case ((p1, f1), (p2, f2)) => (PositionIndex(p1) -> FrameIndex(f1), PositionIndex(p2) -> FrameIndex(f2)) }
    
    def genNonnegativePair: Gen[NNPair] = Gen.zip(genNonnegativeInt, genNonnegativeInt)    

    def getInputFilename(pos: PositionIndex, frame: FrameIndex): String = s"bead_rois__${pos.get}_${frame.get}.csv"
    
    def maxNumRoisSmallTests: ShiftingCount = ShiftingCount(20)

    def writeBundle(folder: os.Path)(pf: PosFramePair, bundle: SmallDataSet.InputBundle): os.Path = {
        val fp = folder / getInputFilename.tupled(pf)
        os.write(fp, bundle.lines.mkString("\n"))
        fp
    }

end TestPartitionIndexedDriftCorrectionRois
