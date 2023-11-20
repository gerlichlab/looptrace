package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ Random, Try }

import cats.data.{ NonEmptyList as NEL }
import cats.syntax.apply.*
import cats.syntax.either.*
import cats.syntax.eq.*
import cats.syntax.list.*
import cats.syntax.option.*
import mouse.boolean.*
import upickle.default.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*

import at.ac.oeaw.imba.gerlich.looptrace.PartitionIndexedDriftCorrectionRois.{
    BadRecord,
    BeadRoisPrefix, 
    ColumnName,
    InitFile,
    ParserConfig, 
    PosFramePair,
    Purpose,
    RoisFileParseFailedRecords,
    RoisFileParseFailedSetup,
    RoisPartition,
    RoiSplitFailure,
    RoisSplitResult,
    RoiSplitSuccess,
    TooFewAccuracyRois, 
    TooFewRois,
    TooFewRoisLike,
    TooFewShiftingRois,
    XColumn, 
    YColumn, 
    ZColumn, 
    discoverInputs, 
    getOutputFilepath,
    getOutputSubfolder,
    readRoisFile,
    sampleDetectedRois, 
    workflow, 
    writeTooFewRois
}
import at.ac.oeaw.imba.gerlich.looptrace.PathHelpers.listPath
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.readJsonFile
import at.ac.oeaw.imba.gerlich.looptrace.space.{ CoordinateSequence, Point3D, XCoordinate, YCoordinate, ZCoordinate }

/** Tests for the partitioning of regions of interest (ROIs) for drift correction */
class TestPartitionIndexedDriftCorrectionRois extends AnyFunSuite, ScalacheckSuite, should.Matchers, PartitionRoisSuite:
    import SelectedRoi.*
    
    /* *******************************************************************************
     * Main test cases                                                                 
     * *******************************************************************************
     */

    test("Parser config roundtrips through JSON.") {
        forAll { (original: ParserConfig) => 
            val jsonData = write(original, indent = 2)
            withTempFile(jsonData){ UJsonHelpers.readJsonFile[ParserConfig](_: os.Path) shouldEqual original }
        }
    }

    test("Trying to read nonexistent parser config file fails expectedly.") {
        withTempDirectory{ (tempdir: os.Path) => 
            val nonextantFile = tempdir / "does_not_exist.json"
            os.exists(nonextantFile) shouldBe false
            assertThrows[ParserConfig.ParseError]{ ParserConfig.readFileUnsafe(nonextantFile) }
        }
    }

    test("Trying to read empty parser config file fails expectedly.") {
        withTempJsonFile("") { (emptyFile: os.Path) => 
            os.exists(emptyFile) shouldBe true
            assertThrows[ParserConfig.ParseError]{ ParserConfig.readFileUnsafe(emptyFile) }
        }
    }

    test("Missing key(s) in config file causes error") {
        /* Generated type loses typelevel information that by construction constrains the domain, 
        so here we don't want shrinking since it will not respect this domain constraint given by the types. */
        implicit def noShrink[A]: Shrink[A] = Shrink.shrinkAny

        /**  Generate a subset of parser config keys to remove and the parser config text-to-text mapping. */
        def genRemovalsAndBase: Gen[(Set[String], Map[String, String])] = for {
            configAdt <- genParserConfig
            configRaw = read[Map[String, String]](write(configAdt))
            keysToRemove <- Gen.someOf(configRaw.keySet)
        } yield (keysToRemove.toSet, configRaw)
        
        forAll (genRemovalsAndBase) { case (toRemove, baseRaw) => 
            val data = baseRaw -- toRemove
            val jsonText = write(data, indent = 2)
            withTempJsonFile(jsonText){ (confFile: os.Path) =>
                // The parse should succeed if and only if...
                if (toRemove.isEmpty) {
                    // ... nothing was removed to the base data...
                    val expected = read[ParserConfig](write(baseRaw))
                    val observed = ParserConfig.readFileUnsafe(confFile)
                    observed shouldEqual expected
                } else {
                    // ...otherwise, the parse should fail.
                    assertThrows[ParserConfig.ParseError]{ ParserConfig.readFileUnsafe(confFile) }
                }
            }
        }
    }

    test("Extra keys in config file causes error -- https://github.com/com-lihaoyi/upickle/issues/537") {
        /* Generated type loses typelevel information that by construction constrains the domain, 
        so here we don't want shrinking since it will not respect this domain constraint given by the types. */
        implicit def noShrink[A]: Shrink[A] = Shrink.shrinkAny

        /**  Generate the parser config text-to-text mapping and values to add. */
        def genBaseAndAdditions: Gen[(Map[String, String], Map[String, String])] = for {
            configRaw <- genParserConfig.map(c => read[Map[String, String]](write(c)))
            additions <- arbitrary[Map[String, String]].suchThat{ m => m.nonEmpty && (m.keySet & configRaw.keySet).isEmpty }
        } yield (configRaw, additions)
        
        pendingUntilFixed {
            forAll (genBaseAndAdditions) { case (baseRaw, additions) => 
                val data = baseRaw ++ additions
                val jsonText = write(data, indent = 2)
                withTempJsonFile(jsonText){ (confFile: os.Path) =>
                    additions.nonEmpty shouldBe true
                    assertThrows[upickle.core.AbortException]{ ParserConfig.readFileUnsafe(confFile) }
                }
            }
        }
    }

    test("Collision between column-like names in parser config is illegal.") {
        /* Build the random input generator for this test, assuring that at least one column name collision occurs */
        def genTextLikes: Gen[(XColumn, YColumn, ZColumn, String)] = {
            def genElems: List[Int] => Gen[List[String]] = structure => {
                val genBlock: Gen[List[List[String]]] = 
                    Gen.sequence(structure map { n => arbitrary[String].map(List.fill(n)) })
                genBlock.map(_.flatten)
            }
            val rawTextValues: Gen[List[String]] = for {
                numUniq <- Gen.oneOf(1, 3)
                structure <- numUniq match {
                    case 1 => Gen.const(List(4))
                    case 2 => Gen.oneOf(List(2, 2), List(1, 3))
                    case 3 => Gen.const(List(1, 1, 2))
                }
                subs <- genElems(structure)
            } yield Random.shuffle(subs)
            rawTextValues.map{
                case x :: y :: z :: qc :: Nil => (XColumn(x), YColumn(y), ZColumn(z), qc)
                case raws => throw new Exception(s"${raws.length} values generated when 4 were expected!")
            }
        }
        
        forAll(Gen.zip(genTextLikes, arbitrary[CoordinateSequence])) { 
            case ((x, y, z, qc), cs) => assertThrows[IllegalArgumentException]{ ParserConfig(x, y, z, qc, cs) }
        }
    }

    test("Illegal type for any parser config value is illegal.") {
        
        implicit def noShrink[A]: Shrink[A] = Shrink.shrinkAny

        /* JSON value for a non-string type */
        def genNonStrJson: Gen[ujson.Value] = Gen.oneOf(
            Gen.const(ujson.Null),
            arbitrary[Int].map(UJsonHelpers.liftInt),
            arbitrary[Double].map(UJsonHelpers.liftDouble), 
            arbitrary[List[Int]].map(ujson.Arr.from),
            arbitrary[List[Double]].map(ujson.Arr.from), 
            arbitrary[Map[String, String]].map(UJsonHelpers.liftMap),
            arbitrary[Map[String, Int]].map(UJsonHelpers.liftMap),
            arbitrary[Map[String, Double]].map(UJsonHelpers.liftMap), 
        )
        
        /* Invalid text to try to parse as coordinate sequence */
        def genCoordseq(using arbStr: Arbitrary[String]): Gen[Either[ujson.Value, CoordinateSequence]] = Gen.oneOf(
            arbStr.suchThat{ s => Try{ read[CoordinateSequence](s) }.isFailure }.gen.map(UJsonHelpers.liftStr), 
            arbitrary[CoordinateSequence]
        ).map((_: ujson.Value | CoordinateSequence) match {
            case v: ujson.Value => Left(v)
            case c: CoordinateSequence => Right(c)
        })

        def toValue = (_: ujson.Value | ColumnName | String | CoordinateSequence) match {
            case v: ujson.Value => v
            case cn: ColumnName => ColumnName.toJson(cn)
            case s: String => UJsonHelpers.liftStr(s)
            case cs: CoordinateSequence => CoordinateSequence.toJson(cs)
        }

        def genCoordinateColumns: Gen[Either[(ujson.Value, ujson.Value, ujson.Value), (XColumn, YColumn, ZColumn)]] =
            Gen.zip(Gen.oneOf(genNonStrJson, arbitrary[XColumn]), Gen.oneOf(genNonStrJson, arbitrary[YColumn]), Gen.oneOf(genNonStrJson, arbitrary[ZColumn])).map{
                case (x: XColumn, y: YColumn, z: ZColumn) => Right((x, y, z)) // Each generated value is legitimate.
                case (x, y, z) => { // At least on generated value is illegitimate.
                    // Make sure we have JSON values, not ADT values.
                    Left(toValue(x), toValue(y), toValue(z))
                }
            }.suchThat{ _ match {
                case Left(_) => true
                case Right(x, y, z) => Set(x.get, y.get, z.get).size === 3 // no column name collision
            } }

        def genQCColumn: Gen[Either[ujson.Value, String]] = Gen.oneOf(genNonStrJson, arbitrary[String]).map((_: ujson.Value | String) match {
            case v: ujson.Value => Left(v)
            case s: String => Right(s)
        })

        def genJsonTextAndExpectation: Gen[(String, Option[ParserConfig])] = 
            Gen.zip(genCoordinateColumns, genQCColumn, genCoordseq)
                .suchThat{
                    case (Right((x, y, z)), Right(qc), _) => qc =!= x.get && qc =!= y.get && qc =!= z.get
                    case _ => true
                }
                .map{
                    case (Right(x, y, z), Right(qc), Right(cs)) => {
                        val exp = ParserConfig(x, y, z, qc, cs)
                        write(exp, indent = 2) -> exp.some
                    }
                    case (xyz, qc, cs) => {
                        val (xVal, yVal, zVal) = xyz.fold(identity, { case (xc, yc, zc) => (toValue(xc), toValue(yc), toValue(zc)) })
                        val qcVal = qc.fold(identity, toValue)
                        val csVal = cs.fold(identity, CoordinateSequence.toJson)
                        val (xKey, yKey, zKey, qcKey, csKey) = labelsOf[ParserConfig]
                        val jsonData = ujson.Obj(
                            ParserConfig.xFieldName -> xVal,
                            ParserConfig.yFieldName -> yVal,
                            ParserConfig.zFieldName -> zVal,
                            ParserConfig.qcFieldName -> qcVal,
                            ParserConfig.csFieldName -> csVal
                        )
                        write(jsonData, indent = 2) -> None
                    }
                }

        forAll (genJsonTextAndExpectation, minSuccessful(10000)) { 
            case (jsonText, expOpt) => expOpt match {
                case None => assertThrows[upickle.core.AbortException | IllegalArgumentException]{ read[ParserConfig](jsonText) }
                case Some(expected) => read[ParserConfig](jsonText) shouldEqual expected
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

    test("ROI request sizes must be positive integers.") {
        assertCompiles("sampleDetectedRois(PositiveInt(1), PositiveInt(1))(List())") // negative control
        
        /* Alternatives with at least 1 positive int */
        assertDoesNotCompile("sampleDetectedRois(PositiveInt(1), NonnegativeInt(1))(List())")
        assertDoesNotCompile("sampleDetectedRois(NonnegativeInt(1), PositiveInt(1))(List())")
        assertDoesNotCompile("sampleDetectedRois(PositiveInt(1), 1)(List())")
        assertDoesNotCompile("sampleDetectedRois(1, PositiveInt(1))(List())")
        
        /* Other alternatives with at least 1 nonnegative int */
        assertDoesNotCompile("sampleDetectedRois(NonnegativeInt(1), NonnegativeInt(1))(List())")
        assertDoesNotCompile("sampleDetectedRois(NonnegativeInt(1), 1)(List())")
        assertDoesNotCompile("sampleDetectedRois(1, NonnegativeInt(1))(List())")
        
        // Alternative with simple integers
        assertDoesNotCompile("sampleDetectedRois(1, 1)(List())")
    }
    
    test("Requesting ROIs count size greater than usable record count yields expected result.") {
        import TooFewRoisLike.problem
        val maxRoisCount = PositiveInt(1000)
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        type Expectation = TooFewShiftingRois | TooFewRois

        def getExpectation(reqShifting: PositiveInt, reqAccuracy: PositiveInt, numUsable: NonnegativeInt): Expectation = {
            if (reqShifting > numUsable) TooFewShiftingRois(reqShifting, numUsable)
            else if (reqShifting + reqAccuracy > numUsable) {
                val expAccRealized = NonnegativeInt.unsafe(numUsable - reqShifting) // guaranteed safe by falsehood of (del > numUsable)
                TooFewRois(reqAccuracy, expAccRealized, Purpose.Accuracy)
            }
            else { throw new IllegalArgumentException(s"Sample size is NOT in excess of usable count: ${reqShifting + reqAccuracy} <= ${numUsable}") }
        }

        final case class InputsAndExpectation(reqShifting: PositiveInt, reqAccuracy: PositiveInt, rois: Iterable[DetectedRoi], expectation: Expectation)

        def genSampleSizeInExcessOfAllRois: Gen[InputsAndExpectation] = for {
            rois <- Gen.choose(0, maxRoisCount).flatMap{ n => Gen.listOfN(n, arbitrary[DetectedRoi]) }
            del <- Gen.choose(1, maxRoisCount).map(PositiveInt.unsafe)
            acc <- Gen.choose(scala.math.max(0, rois.size - del) + 1, maxRoisCount).map(PositiveInt.unsafe)
            exp = getExpectation(del, acc, NonnegativeInt.unsafe(rois.count(_.isUsable)))
        } yield InputsAndExpectation(del, acc, rois, exp)

        def genSampleSizeInExcessOfUsableRois: Gen[InputsAndExpectation] = for {
            numUsable <- Gen.choose(1, maxRoisCount - 1).map(PositiveInt.unsafe)
            usable <- Gen.listOfN(numUsable, arbitrary[DetectedRoi].map(_.copy(isUsable = true)))
            unusable <- Gen.choose(1, maxRoisCount - numUsable).flatMap{ Gen.listOfN(_, arbitrary[DetectedRoi].map(_.copy(isUsable = false))) }
            rois = Random.shuffle(usable ++ unusable)
            del <- Gen.choose(1, numUsable).map(PositiveInt.unsafe)
            acc <- Gen.choose(numUsable - del + 1, rois.size - del).map(PositiveInt.unsafe)
            exp = getExpectation(del, acc, numUsable.asNonnegative)
        } yield InputsAndExpectation(del, acc, rois, exp)
        
        forAll (Gen.oneOf(genSampleSizeInExcessOfAllRois, genSampleSizeInExcessOfUsableRois), minSuccessful(10000)) { 
            case InputsAndExpectation(numShifting, numAccuracy, rois, expectation) => 
                val observation = sampleDetectedRois(numShifting, numAccuracy)(rois)
                (observation, expectation) match {
                    case (obs: TooFewShiftingRois, exp: TooFewShiftingRois) => obs shouldEqual exp
                    case (obs: TooFewAccuracyRois, exp: TooFewRois) => obs.problem shouldEqual exp
                    case _ => fail(s"Incompatible observation ($observation) and expectation ($expectation)")
                }
        }
    }

    test("Empty ROIs file causes expected error.") {
        forAll { (parserConfig: ParserConfig, delimiter: Delimiter) => 
            withTempFile("", delimiter){ (roisFile: os.Path) => 
                os.isFile(roisFile) shouldBe true
                readRoisFile(parserConfig)(roisFile) shouldEqual Left(NEL.one(s"No lines in file! $roisFile"))
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

        forAll (Gen.zip(arbitrary[ParserConfig], genInvalidExt, genHeaderAndGetExtraErrorOpt)) { 
            case (parserConfig, ext, (header, maybeGetExpError)) => 
                withTempFile(initData = header, suffix = ext){ (roisFile: os.Path) => 
                    val extras: List[String] = maybeGetExpError.fold(List())(getMsg => List(getMsg(roisFile)))
                    val expErrorMessages = NEL(s"Cannot infer delimiter for file! $roisFile", extras)
                    readRoisFile(parserConfig)(roisFile) shouldEqual Left(RoisFileParseFailedSetup(expErrorMessages))
                }
        }
    }

    test("Header-only file parses but yields empty record collection.") {
        def createHeaderLine(conf: ParserConfig, delimiter: Delimiter): String =
            delimiter `join` getParserConfigColumnNames(conf).toArray

        forAll(minSuccessful(1000)) { (parserConfig: ParserConfig, delimiter: Delimiter) => 
            val headLine = createHeaderLine(parserConfig, delimiter) ++ "\n"
            withTempFile(headLine, delimiter){ (roisFile: os.Path) => 
                readRoisFile(parserConfig)(roisFile) shouldEqual Right(List())
            }
        } 
    }

    test("Any missing column name in header causes error.") {
        // Create the parser config and a strict subset of the column names.
        def genParserConfigAndHeadFieldSubset: Gen[(ParserConfig, List[String])] = for {
            conf <- arbitrary[ParserConfig]
            minCols = List(conf.xCol.get, conf.yCol.get, conf.zCol.get, conf.qcCol)
            subset <- Gen.choose(0, minCols.length - 1).flatMap(Gen.pick(_, minCols))
        } yield (conf, subset.toList)

        // Optionally, generate some additional column names, limiting to relatively few columns.
        def genParserConfigHeaderAndDelimiter = for {
            (parserConfig, headerSubset) <- genParserConfigAndHeadFieldSubset
            usefulColumns  = getParserConfigColumnNames(parserConfig).toSet
            extras <- Gen.choose(0, 5).flatMap(Gen.listOfN(_, Gen.alphaNumStr.suchThat(!usefulColumns.contains(_))))
            delimiter <- arbitrary[Delimiter]
        } yield (parserConfig, Random.shuffle(headerSubset ::: extras), delimiter)
        
        forAll (genParserConfigHeaderAndDelimiter) { 
            case (parserConfig, headerFields, delimiter) =>
                val expMissFields = getParserConfigColumnNames(parserConfig).toSet -- headerFields.toSet
                val expMessages = expMissFields.map(name => s"Missing field in header: $name")
                val headLine = delimiter.join(headerFields.toArray) ++ "\n"
                withTempFile(headLine, delimiter){ (roisFile: os.Path) => 
                    readRoisFile(parserConfig)(roisFile) match {
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
                    readRoisFile(SmallDataSet.standardParserConfig)(roisFile) match
                        case Right(_) => fail("Parse succeeded when it should've failed!")
                        case Left(bads) => bads shouldEqual RoisFileParseFailedRecords(expBadRecords)
                }
        }
    }

    test("Coordinate sequence has no effect whatsoever on the detected ROI parse.") {
        import SmallDataSet.*

        forAll (Gen.zip(arbitrary[CoordinateSequence].map(standardParserConfig), Gen.oneOf(input1, input2))) {
            case (parserConfig, inputBundle) => 
                val roisText = inputBundle.lines mkString "\n"
                withTempFile(roisText, delimiter) { (roisFile: os.Path) => 
                    val obsFwd = readRoisFile(parserConfig.copy(coordinateSequence = CoordinateSequence.Forward))(roisFile)
                    val obsRev = readRoisFile(parserConfig.copy(coordinateSequence = CoordinateSequence.Reverse))(roisFile)
                    val exp = NonnegativeInt.indexed(inputBundle.points).map{ 
                        case (pt, i) => 
                            val idx = RoiIndex(i)
                            DetectedRoi(idx, pt, inputBundle.partition.usable.contains(idx))
                    }
                    (obsFwd, obsRev) match {
                        case (Right(fwd), Right(rev)) => 
                            fwd shouldEqual exp
                            rev shouldEqual exp
                        case (Left(problems), Right(_)) => fail(s"Forward parse failed! $problems")
                        case (Right(_), Left(problems)) => fail(s"Reverse parse failed! $problems")
                        case (Left(problemsForward), Left(problemsReverse)) => fail(s"Both parses failed! Forward problems: $problemsForward. Reverse problems: $problemsReverse")
                    }
                }
        }
    }

    test("An ROI is never used for more than one purpose.") {
        val maxRoisCount = PositiveInt(1000)
        def genGoodInput: Gen[(PositiveInt, PositiveInt, Iterable[DetectedRoi])] = for {
            numUsable <- Gen.choose(2, maxRoisCount - 1)
            usable <- Gen.listOfN(numUsable, genDetectedRoiFixedUse(true))
            numUnusable <- Gen.choose(1, maxRoisCount - numUsable)
            unusable <- Gen.listOfN(numUnusable, genDetectedRoiFixedUse(false))
            numShifting <- Gen.choose(1, numUsable - 1).map(PositiveInt.unsafe)
            numAccuracy <- Gen.choose(1, maxRoisCount).map(PositiveInt.unsafe)
        } yield (numShifting, numAccuracy, Random.shuffle(usable ++ unusable))
        
        def simplifyRoi(roi: RoiForShifting | RoiForAccuracy): (RoiIndex, Point3D) = roi.index -> roi.centroid

        forAll (genGoodInput, minSuccessful(1000)) { case (numShifting, numAccuracy, rois) => 
            sampleDetectedRois(numShifting, numAccuracy)(rois) match {
                case result: RoiSplitFailure => fail(s"Expected successful partition but got failure: $result")
                case result: RoiSplitSuccess => 
                    val part = result.partition
                    part.shifting.length shouldEqual part.shifting.toSet.size // no duplicates within shifting
                    part.accuracy.length shouldEqual part.accuracy.toSet.size // no duplicates within accuracy
                    (part.shifting.map(simplifyRoi).toSet & part.accuracy.map(simplifyRoi).toSet) shouldEqual Set()
            }
        }
    }

    test("Integration: toggle for tolerance of insufficient shifting ROIs works.") {
        /**
         * In this test, we generate cases in which it's possible that either one or both datasets
         * have sufficient ROI counts for the randomly generated shifting and accuracy ROI counts, 
         * or the one or both of the datasets have insufficient ROIs for the shifting and/or 
         * accuracy requests. The tolerance for insufficient shifting ROIs is also randomised, 
         * and expected output files present, and expected contents, are accordingly adjusted.
        */
        import SmallDataSet.*
        import TooFewAccuracyRois.given
        import TooFewShiftingRois.given

        implicit def noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        def genConfig = arbitrary[CoordinateSequence].map(standardParserConfig)

        type PF = PosFramePair
        val N1 = input1.points.length
        val N2 = input2.points.length

        def genSampleSizes: Gen[(PositiveInt, PositiveInt)] = for {
            numShifting <- Gen.choose(1, scala.math.min(N1, N2) - 1).map(PositiveInt.unsafe)
            numAccuracy <- Gen.choose(1, scala.math.max(N1, N2) + 1).map(PositiveInt.unsafe)
        } yield (numShifting, numAccuracy)
        
        def gen5PF = arbitrary[(PF, PF, PF, PF, PF)].suchThat{ case (a, b, c, d, e) => Set(a, b, c, d, e).size === 5 }
        
        given inputArb: Arbitrary[InputBundle] = Arbitrary{ Gen.oneOf(input1, input2) }
        def genInputsWithPF = Gen.zip(arbitrary[(InputBundle, InputBundle, InputBundle, InputBundle, InputBundle)], gen5PF) map {
            case ((in1, in2, in3, in4, in5), (pf1, pf2, pf3, pf4, pf5)) => ((pf1, in1), (pf2, in2), (pf3, in3), (pf4, in4), (pf5, in5))
        }

        def genInputsAndExpectation = for {
            (numShifting, numAccuracy) <- genSampleSizes
            (a@(pf1, in1), b@(pf2, in2), c@(pf3, in3), d@(pf4, in4), e@(pf5, in5)) <- genInputsWithPF
            usedFrames = Set(pf1._2, pf2._2, pf3._2, pf4._2, pf5._2)
            useOneAsRef <- Gen.oneOf(false, true)
            refFrame <- (
                if useOneAsRef 
                then Gen.oneOf(usedFrames).map(_.some)
                else Gen.option(arbitrary[FrameIndex]).suchThat(_.fold(true)(i => !usedFrames.contains(i)))
                )
            (fatal, nonfatal) = List(a, b, c, d, e).foldRight(List.empty[(PF, InputBundle, TooFewShiftingRois)], List.empty[(PF, InputBundle, TooFewAccuracyRois)]) { 
                case ((pf, in), (worse, bads)) => 
                    if in.numUsable < numShifting then ((pf, in, TooFewShiftingRois(numShifting, in.numUsable)) :: worse, bads)
                    else if in.numUsable < numShifting + numAccuracy then 
                        // dummy null partition here, since it should never be accessed (only care about the requested and realised counts)
                        val err = TooFewAccuracyRois(null, numAccuracy, NonnegativeInt.unsafe(in.numUsable - numShifting))
                        (worse, (pf, in, err) :: bads)
                    else (worse, bads)
                }
            (expError, expSevere) = (fatal, refFrame) match {
                case (Nil, _) => (None, None)
                case (_, None) => (Exception(s"${fatal.size} (position, frame) pairs with problems.\n${fatal.map(t => t._1 -> t._3)}").some, None)
                case (_, Some(rf)) => fatal.partition(_._1._2 === rf) match {
                    case (Nil, tolerated) => (None, tolerated.map(t => t._1 -> t._3).some)
                    case (untolerated, _) => (Exception(s"${untolerated.size} (position, frame) pairs with problems.\n${untolerated.map(t => t._1 -> t._3)}").some, None)
                }
            }
            expWarn = (expError.isEmpty && nonfatal.nonEmpty).option{ nonfatal.map(t => t._1 -> t._3) }
        } yield (List(a, b, c, d, e), numShifting, numAccuracy, refFrame, (expError, expSevere, expWarn))

        forAll (Gen.zip(genConfig, genInputsAndExpectation), minSuccessful(1000)) { 
            case (parserConfig, (inputsWithPF, numShifting, numAccuracy, refFrame, (expError, expSevere, expWarn))) => 
                withTempDirectory{ (tempdir: os.Path) =>
                    inputsWithPF.foreach(writeBundle(tempdir).tupled) // Prep the data.
                    expError match {
                        case Some(exc) => assertThrows[Exception]{ workflow(parserConfig, tempdir, numShifting, numAccuracy, refFrame, None) }
                        case None => 
                            workflow(parserConfig, tempdir, numShifting, numAccuracy, refFrame, None)
                            assertTooFewRoisFileContents(tempdir / "roi_partition_warnings.severe.json", expSevere)
                            assertTooFewRoisFileContents(tempdir / "roi_partition_warnings.json", expWarn)
                    }
                }
        }
    }

    test("Integration: shifting <= #(usable ROIs) < shifting + accuracy ==> warnings file correctly produced; #116") {
        import SmallDataSet.*
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        def genInputs = for {
            parserConfig <- arbitrary[CoordinateSequence].map(standardParserConfig)
            numShifting <- Gen.choose(1, maxRequestNum - 1).map(PositiveInt.unsafe)
            numAccuracy <- Gen.choose(maxRequestNum - numShifting + 1, TooHighRoisNum).map(PositiveInt.unsafe)
            maybeSubfolderName <- Gen.option(Gen.const("temporary_subfolder"))
            pf1 <- arbitrary[PosFramePair]
            pf2 <- arbitrary[PosFramePair].suchThat(_ =!= pf1)
        } yield (parserConfig, numShifting, numAccuracy, maybeSubfolderName, pf1, pf2)

        forAll (genInputs) { case (parserConfig, numShifting, numAccuracy, maybeSubfolderName, pf1, pf2) =>
            withTempDirectory{ (tempdir: os.Path) =>
                /* Setup the inputs. */
                val confFile = tempdir / "parser_config.json"
                os.write(confFile, write(parserConfig, indent = 2))
                List(pf1 -> input1, pf2 -> input2).foreach(writeBundle(tempdir).tupled)
                
                /* Check that the workflow creates the expected warnings file. */
                val warningsFile = tempdir / "roi_partition_warnings.json"
                os.exists(warningsFile) shouldBe false
                workflow(configFile = confFile, inputRoot = tempdir, numShifting = numShifting, numAccuracy = numAccuracy)
                os.isFile(warningsFile) shouldBe true
            }
        }
    }

    test("Integration: golden path's overall behavioral properties are correct.") {
        import SmallDataSet.*
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        /** Generate shifting and accuracy counts that should yield no warnings and no errors. */
        def genInputs = for {
            coordseq <- arbitrary[CoordinateSequence]
            numShifting <- Gen.choose(1, maxRequestNum - 1).map(PositiveInt.unsafe)
            numAccuracy <- Gen.choose(1, maxRequestNum - numShifting).map(PositiveInt.unsafe)
            maybeSubfolderName <- Gen.option(Gen.const("temporary_subfolder"))
            pf1 <- arbitrary[PosFramePair]
            pf2 <- arbitrary[PosFramePair].suchThat(_ =!= pf1)
        } yield (coordseq, numShifting, numAccuracy, maybeSubfolderName, pf1, pf2)
        
        forAll (genInputs) {
            case (coordseq, numShifting, numAccuracy, maybeSubfolderName, pf1, pf2) => 
                val parserConfig = standardParserConfig(coordseq)
                withTempDirectory{ (tempdir: os.Path) =>
                    /* Setup the inputs. */
                    val confFileName = "parser_config.json"
                    val confFile = tempdir / confFileName
                    os.write(confFile, write(parserConfig, indent = 2))
                    val roisFolder = maybeSubfolderName.fold(tempdir)(tempdir / _)
                    os.makeDir.all(roisFolder)
                    val roiFileParts = List(pf1 -> input1, pf2 -> input2).map { 
                        case (pf, inputBundle) => 
                            val fp = roisFolder / getInputFilename.tupled(pf)
                            os.write(fp, inputBundle.lines.mkString("\n"))
                            pf -> inputBundle.partition
                        }.toMap
                    
                    workflow(configFile = confFile, inputRoot = roisFolder, numShifting = numShifting, numAccuracy = numAccuracy)
                    val shiftingOutfolder = getOutputSubfolder(roisFolder)(Purpose.Shifting)
                    val accuracyOutfolder = getOutputSubfolder(roisFolder)(Purpose.Accuracy)
                    List(shiftingOutfolder, accuracyOutfolder).forall(os.isDir) shouldBe true
                    
                    val posFramePairs = List(pf1, pf2)
                    val expShiftingOutfiles = posFramePairs.map{ case (p, f) => (p, f) -> getOutputFilepath(roisFolder)(p, f, Purpose.Shifting) }
                    val expAccuracyOutfiles = posFramePairs.map{ case (p, f) => (p, f) -> getOutputFilepath(roisFolder)(p, f, Purpose.Accuracy) }
                    given shiftingRoiRW: ReadWriter[RoiForShifting] = SelectedRoi.simpleShiftingRW(coordseq)
                    given accuracyRoiRW: ReadWriter[RoiForAccuracy] = SelectedRoi.simpleAccuracyRW(coordseq)
                    val obsShiftings = expShiftingOutfiles.map { case (pf, fp) => (pf, readJsonFile[List[RoiForShifting]](fp)) }.toMap
                    val obsAccuracies = expAccuracyOutfiles.map { case (pf, fp) => pf -> readJsonFile[List[RoiForAccuracy]](fp) }.toMap
                    
                    val shiftingOutFiles = os.list(shiftingOutfolder)
                    val accuracyOutFiles = os.list(accuracyOutfolder)
                    shiftingOutFiles.length shouldBe posFramePairs.length
                    accuracyOutFiles.length shouldBe posFramePairs.length
                    shiftingOutFiles.toSet shouldEqual expShiftingOutfiles.map(_._2).toSet
                    accuracyOutFiles.toSet shouldEqual expAccuracyOutfiles.map(_._2).toSet
                    obsShiftings.keySet shouldEqual obsAccuracies.keySet
                    obsShiftings.forall(_._2.length === numShifting) shouldBe true
                    obsAccuracies.forall(_._2.length === numAccuracy) shouldBe true
                    
                    val byPosFrame = obsShiftings.map{ case (pf, shifting) => pf -> (shifting, obsAccuracies(pf)) }
                    
                    // Set of ROIs for shifting must have no overlap with ROIs for accuracy.
                    byPosFrame.values.map{ case (shifting, accuracy) => 
                        (shifting.map(_.index).toSet & accuracy.map(_.index).toSet) 
                    } shouldEqual List.fill(posFramePairs.length)(Set()) // no intersection
                    
                    // Each selected ROI (shifting and accuracy) should have been in the usable pool, not unusable.
                    byPosFrame.toList.map{ case (pf, (shifting, accuracy)) => 
                        val delIdx = shifting.map(_.index).toSet
                        val accIdx = accuracy.map(_.index).toSet
                        val part = roiFileParts(pf)
                        (delIdx.forall(part.usable.contains), !delIdx.exists(part.unusable.contains), accIdx.forall(part.usable.contains), !accIdx.exists(part.unusable.contains))
                    } shouldEqual List.fill(posFramePairs.length)((true, true, true, true))
                    
                    // The coordinates of the ROI centroids must be preserved during the selection/partitioning.
                    byPosFrame.toList.map{ case (pf, (shifting, accuracy)) => 
                        val getPoint = roiFileParts(pf).getPointSafe
                        shifting.filterNot(roi => roi.centroid.some === getPoint(roi.index)) -> accuracy.filterNot(roi => roi.centroid.some === getPoint(roi.index))
                    } shouldEqual List.fill(posFramePairs.length)(List() -> List())

                    // There should be no warnings, anywhere.
                    PathHelpers.listPath(tempdir).filter(_.last === "roi_partition_warnings.json").isEmpty shouldBe true
                }
        }
    }

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
        def standardParserConfig: ParserConfig = standardParserConfig(CoordinateSequence.Reverse)
        def standardParserConfig(coordseq: CoordinateSequence) = read[ParserConfig](standardConfigText(coordseq))
        def standardConfigText(coordseq: CoordinateSequence) = s"""
        {
            "xCol": "centroid-0",
            "yCol": "centroid-1",
            "zCol": "centroid-2",
            "qcCol": "fail_code",
            "coordinateSequence": "${coordseq.toString}"
        }
        """
        private def buildPoint(x: Double, y: Double, z: Double) = Point3D(XCoordinate(x), YCoordinate(y), ZCoordinate(z))
    end SmallDataSet

    /* *******************************************************************************
     * Ancillary types and functions
     * *******************************************************************************
     */    
    type NNPair = (NonnegativeInt, NonnegativeInt)

    def assertTooFewRoisFileContents[A](filepath: os.Path, expected: Option[List[(PosFramePair, A)]])(using ev: TooFewRoisLike[A]) = {
        import TooFewRoisLike.* 
        expected match {
            case None => os.exists(filepath) shouldBe false
            case Some(pfTooFewPairs) => 
                os.isFile(filepath) shouldBe true
                val obs = readJsonFile[List[(PosFramePair, TooFewRois)]](filepath)
                val exp = pfTooFewPairs.map{ case (pf, tooFew) => pf -> tooFew.problem }
                obs.length shouldEqual exp.length
                obs.toSet shouldEqual exp.toSet
        }
    }

    def genDistinctNonnegativePairs: Gen[(PosFramePair, PosFramePair)] = 
        Gen.zip(genNonnegativePair, genNonnegativePair)
            .suchThat{ case (p1, p2) => p1 =!= p2 }
            .map { case ((p1, f1), (p2, f2)) => (PositionIndex(p1) -> FrameIndex(f1), PositionIndex(p2) -> FrameIndex(f2)) }
    
    def genNonnegativePair: Gen[NNPair] = Gen.zip(genNonnegativeInt, genNonnegativeInt)    

    def getInputFilename(pos: PositionIndex, frame: FrameIndex): String = s"bead_rois__${pos.get}_${frame.get}.csv"
    
    def getParserConfigColumnNames(conf: ParserConfig): List[String] = List(conf.xCol.get, conf.yCol.get, conf.zCol.get, conf.qcCol)

    def writeBundle(folder: os.Path)(pf: PosFramePair, bundle: SmallDataSet.InputBundle): os.Path = {
        val fp = folder / getInputFilename.tupled(pf)
        os.write(fp, bundle.lines.mkString("\n"))
        fp
    }

end TestPartitionIndexedDriftCorrectionRois
