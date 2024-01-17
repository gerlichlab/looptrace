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

/** Tests for the simple pairwise distances computation program. */
class TestComputeSimpleDistances extends AnyFunSuite, LooptraceSuite, ScalacheckSuite, should.Matchers:
    
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
                val expError = Input.UnexpectedHeaderException(observed = augmentedHeader, expected = Input.allColumns)
                obsError shouldEqual expError
            }
        }
    }

    test("Trying to use a file with just records and no header fails the parse as expected.") { pending }

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
                val expError = Input.UnexpectedHeaderException(observed = expectedHeader, expected = Input.allColumns)
                obsError.expected shouldEqual expError.expected
                obsError.observed shouldEqual expError.observed
                obsError shouldEqual expError
            }
        }
    }

    test("Simply permuting the header fields fails the parse with the expected error.") { pending } 

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
            def genBadLocus: Gen[Mutate] = genMutate(Input.LocusSpecificBarcodeTimepointColun, Gen.oneOf(Gen.alphaStr, arbitrary[Double]))
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
    
    test("Distances computed are accurately Euclidean.") { pending }
    
    test("Completeness: All pairs of distances are computed within each grouping unit.") { pending }
    
    test("Specificity: Pairs of rows in different groups of (FOV, region, trace) never have distance computed.") { pending }

    /* Instance for random case / example generation */
    given arbitraryForTraceId(using arbRoiIdx: Arbitrary[RoiIndex]): Arbitrary[TraceId] = arbRoiIdx.map(TraceId.fromRoiIndex)
    given arbitraryForGroupName(using arbTime: Arbitrary[FrameIndex]): Arbitrary[GroupName] = arbTime.map(GroupName.fromFrameIndex)
    given arbitraryForGoodInputReceord(using 
        arbPos: Arbitrary[PositionIndex], 
        arbTrace: Arbitrary[TraceId], 
        arbRegion: Arbitrary[GroupName], 
        arbLocus: Arbitrary[FrameIndex], 
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
        List(r.position.show, r.trace.show, r.region.show, r.frame.show, x.get.show, y.get.show, z.get.show)
    }
end TestComputeSimpleDistances
