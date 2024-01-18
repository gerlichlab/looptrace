package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.*
import cats.syntax.all.*
import mouse.boolean.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*

import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.safeReadAllWithOrderedHeaders
import at.ac.oeaw.imba.gerlich.looptrace.ComputeSimpleDistances.*
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.ComputeSimpleDistances.Input.getGroupingKey

/** Tests for the simple pairwise distances computation program. */
class TestComputeSimpleDistances extends AnyFunSuite, LooptraceSuite, ScalacheckSuite, should.Matchers:
    
    test("Totally empty input file causes expected error.") {
        withTempDirectory{ (tempdir: os.Path) => 
            /* Setup and pretests */
            val infile = tempdir / "input.csv"
            val outfolder = tempdir / "output"
            os.makeDir(outfolder)
            touchFile(infile)
            os.isDir(outfolder) shouldBe true
            os.isFile(infile) shouldBe true
            assertThrows[Input.EmptyFileException]{ workflow(inputFile = infile, outputFolder = outfolder) }
        }
    }

    test("Input file that's just a header produces an output file that's just a header.") {
        withTempDirectory{ (tempdir: os.Path) => 
            /* Setup and pretests */
            val infile = tempdir / "input.csv"
            val outfolder = tempdir / "output"
            os.makeDir(outfolder)
            os.write(infile, Delimiter.CommaSeparator.join(Input.allColumns) ++ "\n")
            val expOutfile = outfolder / "input.pairwise_distances.csv"
            os.exists(expOutfile) shouldBe false
            workflow(inputFile = infile, outputFolder = outfolder)
            os.isFile(expOutfile) shouldBe true
            safeReadAllWithOrderedHeaders(expOutfile) match {
                case Left(err) => fail(s"Expected successful output file parse but got error: $err")
                case Right((header, _)) => header shouldEqual OutputWriter.header
            }
        }
    }

    test("Trying to use a file with just records and no header fails the parse as expected.") {
        forAll { (records: NonEmptyList[Input.GoodRecord]) => 
            withTempDirectory{ (tempdir: os.Path) => 
                val infile = tempdir / "input.csv"
                os.write(infile, records.toList.map(recordToTextFields `andThen` rowToLine))
                val expError = Input.UnexpectedHeaderException(recordToTextFields(records.head))
                val obsError = intercept[Input.UnexpectedHeaderException]{ workflow(inputFile = infile, outputFolder = tempdir / "output") }
                obsError shouldEqual expError
            }
        }
    }

    test("Unexpected header error can't be created with the expected input header.") {
        val error = intercept[IllegalArgumentException]{ Input.UnexpectedHeaderException(Input.allColumns) }
        error.getMessage shouldEqual s"requirement failed: Alleged inequality between observed and expected header, but they're equivalent!"
    }

    test("Simply permuting the header fields fails the parse with the expected error.") {
        def genBadHeader = Gen.oneOf(Input.allColumns.permutations.toSeq).suchThat(_ =!= Input.allColumns)
        forAll (arbitrary[NonEmptyList[Input.GoodRecord]], genBadHeader) { (records, mutantHeader) =>
            withTempDirectory{ (tempdir: os.Path) => 
                val infile = tempdir / "input.csv"
                val expError = Input.UnexpectedHeaderException(mutantHeader)
                writeMinimalInputCsv(infile, mutantHeader :: records.toList.map(recordToTextFields))
                val obsError = intercept[Input.UnexpectedHeaderException]{ workflow(inputFile = infile, outputFolder = tempdir / "output") }
                obsError shouldEqual expError
            }
        }
    }

    test("Any nonempty subset of missing/incorrect columns from input file causes expected error.") {
        type ExpectedHeader = List[String]
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        
        given arbExpHeader: Arbitrary[ExpectedHeader] = {
            def genDeletions: Gen[ExpectedHeader] = Gen.choose(1, Input.allColumns.length - 1)
                .flatMap(Gen.pick(_, (0 until Input.allColumns.length)))
                .map{ indices => Input.allColumns.zipWithIndex.filterNot{ (_, i) => indices.toSet contains i }.map(_._1) }
            def genSubstitutions: Gen[ExpectedHeader] = for {
                indicesToChange <- Gen.atLeastOne((0 until Input.allColumns.length)).map(_.toSet) // Choose header fields to change.
                expectedHeader <- Input.allColumns.zipWithIndex.traverse{ (col, idx) => 
                    if indicesToChange contains idx 
                    then Gen.alphaNumStr.suchThat(_ =!= col) // Ensure the replacement differs from original.
                    else Gen.const(col) // Use the original value since this index isn't one at which to update.
                }
            } yield expectedHeader
            
            Gen.oneOf(genSubstitutions, genDeletions).toArbitrary
        }
        
        forAll { (expectedHeader: ExpectedHeader, records: NonEmptyList[Input.GoodRecord]) =>
            withTempDirectory{ (tempdir: os.Path) => 
                val infile = {
                    val f = tempdir / "input.csv"
                    os.write(f, (expectedHeader :: records.toList.map(recordToTextFields)).map(rowToLine))
                    f
                }
                val obsError = intercept[Input.UnexpectedHeaderException]{ workflow(inputFile = infile, outputFolder = tempdir / "output") }
                val expError = Input.UnexpectedHeaderException(expectedHeader)
                obsError shouldEqual expError
            }
        }
    }

    test("Any row with too few fields, too many fields, or improperly typed value(s) breaks the parse.") {
        type Mutate = List[String] => List[String]
        given showForStringOrDouble: Show[String | Double] = Show.show((_: String | Double) match {
            case x: Double => x.toString
            case s: String => s
        }) // used for .show'ing a generated value of more than one type possibility
        
        def genDrops: Gen[Mutate] = Gen.atLeastOne((0 until Input.allColumns.length)).map {
            indices => { (_: List[String]).zipWithIndex.filterNot((_, i) => indices.toSet.contains(i)).map(_._1) }
        }
        def genAdditions: Gen[Mutate] = 
            Gen.resize(5, Gen.nonEmptyListOf(Gen.choose(-3e-3, 3e3).map(_.toString))).map(additions => (_: List[String]) ++ additions)
        def genImproperlyTyped: Gen[Mutate] = {
            def genMutate[A : Show](col: String, alt: Gen[A]): Gen[Mutate] = {
                val idx = Input.allColumns.zipWithIndex.find(_._1 === col).map(_._2).getOrElse{
                    throw new Exception(s"Cannot find index for alleged input column: $col")
                }
                alt.map(a => { (_: List[String]).updated(idx, a.show) })
            }
            def genBadPosition: Gen[Mutate] = genMutate(Input.FieldOfViewColumn, Gen.oneOf(Gen.alphaStr, arbitrary[Double]))
            def genBadTrace: Gen[Mutate] = genMutate(Input.TraceIdColumn, Gen.oneOf(Gen.alphaStr, arbitrary[Double]))
            // NB: Skipping bad region b/c so long as it's String-ly typed, there's no way to generate a bad value.
            def genBadLocus: Gen[Mutate] = genMutate(Input.LocusSpecificBarcodeTimepointColumn, Gen.oneOf(Gen.alphaStr, arbitrary[Double]))
            def genBadPoint: Gen[Mutate] = 
                Gen.oneOf(Input.XCoordinateColumn, Input.YCoordinateColumn, Input.ZCoordinateColumn).flatMap(genMutate(_, Gen.alphaStr))
            Gen.oneOf(genBadPoint, genBadLocus, genBadTrace, genBadPosition)
        }
        
        def genBadRecords: Gen[(NonEmptyList[Int], NonEmptyList[List[String]])] = for {
            goods <- arbitrary[NonEmptyList[Input.GoodRecord]].map(_.map(recordToTextFields))
            indices <- Gen.atLeastOne((0 until goods.length)).map(_.toList.sorted.toNel.get)
            mutations <- indices.toList
                // Put genImproperlyTyped first since there are more possbilities there, 
                // and the distribution is biased towards selection of elements earlier in the sequence.
                .traverse{ i => Gen.oneOf(genImproperlyTyped, genAdditions, genDrops).map(i -> _) }
                .map(_.toMap)
        } yield (indices, goods.zipWithIndex.map{ (r, i) => mutations.get(i).fold(r)(_(r)) })
        
        forAll (genBadRecords) { (expBadRows, textRecords) =>
            withTempDirectory{ (tempdir: os.Path) => 
                val infile = tempdir / "input.csv"
                os.isFile(infile) shouldBe false
                writeMinimalInputCsv(infile, Input.allColumns :: textRecords.toList)
                os.isFile(infile) shouldBe true
                val error = intercept[Input.BadRecordsException]{ workflow(inputFile = infile, outputFolder = tempdir / "output") }
                error.records.map(_.lineNumber) shouldEqual expBadRows
            }
        }
    }

    test("Pandas format is not accepted and causes Input.UnexpectedHeaderException.") {
        val augmentedHeader = "" :: Input.allColumns
        forAll { (records: NonEmptyList[Input.GoodRecord]) => 
            withTempDirectory{ (tempdir: os.Path) => 
                /* Setup and pretests */
                val infile = {
                    val f = tempdir / "input.csv"
                    val indexedTextRows = NonnegativeInt.indexed(records.toList).map((r, i) => i.show :: recordToTextFields(r))
                    os.write(f, (augmentedHeader :: indexedTextRows).map(rowToLine))
                    f
                }
                os.isFile(infile) shouldBe true
                val obsError = intercept[Input.UnexpectedHeaderException]{ workflow(inputFile = infile, outputFolder = tempdir / "output") }
                val expError = Input.UnexpectedHeaderException(augmentedHeader)
                obsError shouldEqual expError
            }
        }
    }

    test("Output file always has the right header") {
        forAll { (records: NonEmptyList[Input.GoodRecord]) => 
            withTempDirectory{ (tempdir: os.Path) => 
                val infile = tempdir / "input_with_suffix.some.random.extensions.csv"
                writeMinimalInputCsv(infile, records)
                val outfolder = tempdir / "output"
                val expOutfile = outfolder / "input_with_suffix.pairwise_distances.csv"
                os.exists(expOutfile) shouldBe false
                workflow(inputFile = infile, outputFolder = outfolder)
                os.isFile(expOutfile) shouldBe true
                os.read.lines(expOutfile).toList match {
                    case Nil => fail("No lines in output file!")
                    case h :: _ => (Delimiter.CommaSeparator `split` h) shouldEqual OutputWriter.header
                }
            }
        }
    }

    test("Distances computed are accurately Euclidean.") {
        def buildPoint(x: Double, y: Double, z: Double) = Point3D(XCoordinate(x), YCoordinate(y), ZCoordinate(z))
        val pos = PositionIndex(NonnegativeInt(0))
        val tid = TraceId(NonnegativeInt(1))
        val reg = GroupName("40")
        val inputRecords = NonnegativeInt.indexed(List((2.0, 1.0, -1.0), (1.0, 5.0, 0.0), (3.0, 0.0, 2.0))).map{
            (pt, i) => Input.GoodRecord(pos, tid, reg, Timepoint(i), buildPoint.tupled(pt))
        }
        val getExpEuclDist = (i: Int, j: Int) => EuclideanDistance.between(inputRecords(i).point, inputRecords(j).point)
        val expected: Iterable[OutputRecord] = List(0 -> 1, 0 -> 2, 1 -> 2).map{ (i, j) => 
            val t1 = Timepoint.unsafe(i)
            val t2 = Timepoint.unsafe(j)
            OutputRecord(pos, tid, reg, t1, t2, getExpEuclDist(i, j), t1.get, t2.get)
        }
        val observed = inputRecordsToOutputRecords(NonnegativeInt.indexed(inputRecords))
        observed shouldEqual expected
    }
    
    test("Partitioning and completeness: (FOV, trace, region) defines an equivalence relation on the set of output records.") {
        def buildPoint(x: Double, y: Double, z: Double) = Point3D(XCoordinate(x), YCoordinate(y), ZCoordinate(z))
        val pt1 = buildPoint(-1.0, 2.0, 0.0)
        val pt2 = buildPoint(1.0, 5.0, 4.0)
        val pt3 = buildPoint(3.0, 4.0, 6.0)
        val pt4 = buildPoint(-2.0, -3.0, 1.0)
        val lonePosition = 1 -> (PositionIndex(NonnegativeInt(2)), TraceId(NonnegativeInt(1)))
        val pairPosition = 2 -> (PositionIndex(NonnegativeInt(5)), TraceId(NonnegativeInt(4)))
        val trioPosition = 3 -> (PositionIndex(NonnegativeInt(7)), TraceId(NonnegativeInt(2)))
        val inputTable = Table(("groupingKeys", "simplifiedExpectation"), List(
            ((1, 1, 1), (2, 1, 1), (2, 1, 2), (1, 1, 1)) -> List((1, 1, "1", 0, 3) -> math.sqrt(27)), 
            ((2, 1, 1), (2, 1, 1), (2, 1, 2), (2, 1, 2)) -> List((2, 1, "1", 0, 1) -> math.sqrt(29), (2, 1, "2", 2, 3) -> math.sqrt(99)), 
            ((2, 1, 1), (2, 1, 2), (2, 1, 1), (2, 1, 2)) -> List((2, 1, "1", 0, 2) -> math.sqrt(56), (2, 1, "2", 1, 3) -> math.sqrt(82)), 
            ((3, 1, 2), (0, 4, 5), (3, 1, 2), (3, 1, 2)) -> List((3, 1, "2", 0, 2) -> math.sqrt(56), (3, 1, "2", 0, 3) -> math.sqrt(27), (3, 1, "2", 2, 3) -> math.sqrt(99)), 
            ((3, 1, 2), (3, 1, 0), (3, 1, 3), (3, 1, 1)) -> List(), // All equal on 1st 2 elements, but not 3rd
            ((0, 1, 2), (3, 1, 2), (1, 1, 2), (2, 1, 2)) -> List(), // All equal on 2nd 2 elements, but not 1st
            ((0, 1, 2), (0, 0, 2), (0, 2, 2), (0, 3, 2)) -> List(), // All equal on 1st and 3rd elements, but not 2nd
            ((0, 1, 2), (1, 1, 2), (0, 2, 2), (2, 2, 2)) -> List(), // Mixed similarities
        )*)
        forAll (inputTable) { case ((k1, k2, k3, k4), expectation) => 
            val records = NonnegativeInt.indexed(List(k1 -> pt1, k2 -> pt2, k3 -> pt3, k4 -> pt4)).map{ 
                case (((pos, tid, reg), pt), i) => Input.GoodRecord(
                    PositionIndex.unsafe(pos), 
                    TraceId(NonnegativeInt.unsafe(tid)), 
                    GroupName(reg.toString), 
                    Timepoint(i), 
                    pt,
                    )
            }
            val observation = inputRecordsToOutputRecords(NonnegativeInt.indexed(records))
            val simplifiedObservation = observation.map{ r => 
                (r.position.get, r.trace.get, r.region.get, r.time1.get, r.time2.get) -> 
                r.distance.get
            }.toList
            val simplifiedExpectation = expectation.map{ case ((pos, tid, reg, t1, t2), d) => 
                (NonnegativeInt.unsafe(pos), NonnegativeInt.unsafe(tid), reg, NonnegativeInt.unsafe(t1), NonnegativeInt.unsafe(t2)) -> 
                NonnegativeReal.unsafe(d)
            }.toList
            // We're indifferent to the order of output records for this test, so check size then convert to maps.
            simplifiedObservation.length shouldEqual simplifiedExpectation.length
            simplifiedObservation.groupBy(_._1).view.mapValues(_.length).toMap shouldEqual simplifiedExpectation.groupBy(_._1).view.mapValues(_.length).toMap
            simplifiedObservation.groupBy(_._1).view.mapValues(_.toSet).toMap shouldEqual simplifiedExpectation.groupBy(_._1).view.mapValues(_.toSet).toMap
        }
    }

    test("When no input records share identical grouping elements, there's never any output.") {
        def genRecords: Gen[NonEmptyList[Input.GoodRecord]] = 
            arbitrary[NonEmptyList[Input.GoodRecord]].suchThat{ rs => 
                rs.length > 1 && 
                rs.toList.combinations(2).forall{
                    case r1 :: r2 :: Nil => Input.getGroupingKey(r1) =!= Input.getGroupingKey(r2)
                    case recs => throw new Exception(s"Got list of ${recs.length} (not 2) when taking pairs!")
                }
            }
        forAll (genRecords) { records => inputRecordsToOutputRecords(NonnegativeInt.indexed(records.toList)).isEmpty shouldBe true }
    }

    test("Any output record's original record indices map them back to input records with identical grouping elements.") {
        /* To encourage collisions, narrow the choices for grouping components. */
        given arbPos: Arbitrary[PositionIndex] = Gen.oneOf(0, 1).map(PositionIndex.unsafe).toArbitrary
        given arbTrace: Arbitrary[TraceId] = Gen.oneOf(2, 3).map(NonnegativeInt.unsafe `andThen` TraceId.apply).toArbitrary
        given arbRegion: Arbitrary[GroupName] = Gen.oneOf(40, 41).map(r => GroupName(r.toString)).toArbitrary
        forAll (Gen.choose(10, 100).flatMap(Gen.listOfN(_, arbitrary[Input.GoodRecord]))) { (records: List[Input.GoodRecord]) => 
            val indexedRecords = NonnegativeInt.indexed(records)
            val getKey = indexedRecords.map(_.swap).toMap.apply.andThen(Input.getGroupingKey)
            val observed = inputRecordsToOutputRecords(indexedRecords)
            observed.filter{ r => getKey(r.inputIndex1) === getKey(r.inputIndex2) } shouldEqual observed
        }
    }

    test("Distance is never computed between records with identically-valued locus-specific timepoints, even if the grouping elements place them together.") {
        /* To encourage collisions, narrow the choices for grouping components. */
        given arbPos: Arbitrary[PositionIndex] = Gen.oneOf(0, 1).map(PositionIndex.unsafe).toArbitrary
        given arbTrace: Arbitrary[TraceId] = Gen.oneOf(2, 3).map(NonnegativeInt.unsafe `andThen` TraceId.apply).toArbitrary
        given arbRegion: Arbitrary[GroupName] = Gen.oneOf(40, 41).map(r => GroupName(r.toString)).toArbitrary
        given arbTime: Arbitrary[Timepoint] = Gen.const(Timepoint(NonnegativeInt(10))).toArbitrary
        forAll (Gen.choose(10, 100).flatMap(Gen.listOfN(_, arbitrary[Input.GoodRecord]))) {
            (records: List[Input.GoodRecord]) => inputRecordsToOutputRecords(NonnegativeInt.indexed(records)).toList shouldEqual List()
        }
    }

    /* Instance for random case / example generation */
    given arbitraryForTraceId(using arbRoiIdx: Arbitrary[RoiIndex]): Arbitrary[TraceId] = arbRoiIdx.map(TraceId.fromRoiIndex)
    given arbitraryForGroupName(using arbTime: Arbitrary[Timepoint]): Arbitrary[GroupName] = arbTime.map(GroupName.fromTimepoint)
    given arbitraryForGoodInputRecord(using 
        arbPos: Arbitrary[PositionIndex], 
        arbTrace: Arbitrary[TraceId], 
        arbRegion: Arbitrary[GroupName], 
        arbLocus: Arbitrary[Timepoint], 
        arbPoint: Arbitrary[Point3D], 
    ): Arbitrary[Input.GoodRecord] = (arbPos, arbTrace, arbRegion, arbLocus, arbPoint).mapN(Input.GoodRecord.apply)

    /** Write the given rows as CSV to the given file. */
    private def writeMinimalInputCsv(f: os.Path, rows: List[List[String]]): Unit = 
        os.write(f, rows map rowToLine)

    /** Write the given records as CSV to the given file. */
    private def writeMinimalInputCsv(f: os.Path, records: NonEmptyList[Input.GoodRecord]): Unit = 
        writeMinimalInputCsv(f, Input.allColumns :: records.toList.map(recordToTextFields))

    /** Convert a sequence of text fields into a single line (CSV), including newline. */
    private def rowToLine = Delimiter.CommaSeparator.join(_: List[String]) ++ "\n"

    /** Convert each ADT value to a simple sequence of text fields, for writing to format like CSV. */
    private def recordToTextFields = (r: Input.GoodRecord) => {
        val (x, y, z) = (r.point.x, r.point.y, r.point.z)
        List(r.position.show, r.trace.show, r.region.show, r.time.show, x.get.show, y.get.show, z.get.show)
    }
end TestComputeSimpleDistances
