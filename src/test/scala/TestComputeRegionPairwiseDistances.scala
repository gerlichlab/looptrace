package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.*
import cats.syntax.all.*
import mouse.boolean.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import io.github.iltotore.iron.scalacheck.char.given

import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanDistance
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.imaging.PositionName
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.ComputeRegionPairwiseDistances.*
import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.safeReadAllWithOrderedHeaders
import at.ac.oeaw.imba.gerlich.looptrace.collections.*
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/**
 * Tests for the simple pairwise distances computation program, for regional barcode spots
 *
 * @author Vince Reuter
 */
class TestComputeRegionPairwiseDistances extends AnyFunSuite, ScalaCheckPropertyChecks, LooptraceSuite, should.Matchers:
    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)
    
    val AllReqdColumns = List(
        Input.FieldOfViewColumn, 
        Input.RegionalBarcodeTimepointColumn, 
        Input.XCoordinateColumn, 
        Input.YCoordinateColumn, 
        Input.ZCoordinateColumn,
        )

    test("Totally empty input file causes expected error.") {
        withTempDirectory{ (tempdir: os.Path) => 
            /* Setup and pretests */
            val infile = tempdir / "input.csv"
            val outfolder = tempdir / "output"
            os.makeDir(outfolder)
            touchFile(infile)
            os.isDir(outfolder) shouldBe true
            os.isFile(infile) shouldBe true
            assertThrows[EmptyFileException]{ workflow(inputFile = infile, outputFolder = outfolder) }
        }
    }

    test("Input file that's just a header produces an output file that's just a header.") {
        forAll(Table("includeIndexCol", List(false, true)*)) { includeIndexCol => 
            withTempDirectory{ (tempdir: os.Path) => 
                /* Setup and pretests */
                val infile = tempdir / "input.csv"
                val outfolder = tempdir / "output"
                os.makeDir(outfolder)
                val cols = if includeIndexCol then "" :: AllReqdColumns else AllReqdColumns
                os.write(infile, Delimiter.CommaSeparator.join(cols) ++ "\n")
                val expOutfile = outfolder / "input.pairwise_distances__regional.csv"
                os.exists(expOutfile) shouldBe false
                workflow(inputFile = infile, outputFolder = outfolder)
                os.isFile(expOutfile) shouldBe true
                safeReadAllWithOrderedHeaders(expOutfile) match {
                    case Left(err) => fail(s"Expected successful output file parse but got error: $err")
                    case Right((header, _)) => header shouldEqual OutputWriter.header
                }
            }
        }
    }

    test("Trying to use a file with just records and no header fails the parse as expected.") {
        forAll { (records: NonEmptyList[Input.GoodRecord], includeIndexCol: Boolean) => 
            withTempDirectory{ (tempdir: os.Path) => 
                val infile = tempdir / "input.csv"
                val toTextFields: (Input.GoodRecord, Int) => List[String] = 
                    if includeIndexCol 
                    then { (r, i) => i.show_ :: recordToTextFields(r) }
                    else { (r, _) => recordToTextFields(r) }
                os.write(infile, records.toList.zipWithIndex.map(toTextFields.tupled `andThen` textFieldsToLine))
                intercept[IllegalHeaderException]{ workflow(inputFile = infile, outputFolder = tempdir / "output") } match {
                    case IllegalHeaderException(obsHead, missing) => 
                        obsHead shouldEqual toTextFields(records.head, 0)
                        // Account for the fact that randomly drawn first-row elements could collide with 
                        // a required header field and therefore reduce the theoretically missing set.
                        missing.some shouldEqual (AllReqdColumns.toNel.get.toNes -- obsHead.toNel.get.toNes).toNonEmptySet
                }
            }
        }
    }

    test("Any nonempty subset of missing/incorrect columns from input file causes expected error.") {
        type ExpectedHeader = List[String]
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        
        given arbExpHeader: Arbitrary[ExpectedHeader] = {
            def genDeletions: Gen[ExpectedHeader] = Gen.choose(1, AllReqdColumns.length - 1)
                .flatMap(Gen.pick(_, (0 until AllReqdColumns.length)))
                .map{ indices => AllReqdColumns.zipWithIndex.filterNot{ (_, i) => indices.toSet contains i }.map(_._1) }
            def genSubstitutions: Gen[ExpectedHeader] = for {
                indicesToChange <- Gen.atLeastOne((0 until AllReqdColumns.length)).map(_.toSet) // Choose header fields to change.
                expectedHeader <- AllReqdColumns.zipWithIndex.traverse{ (oldCol, idx) => 
                    if indicesToChange contains idx 
                    then Gen.alphaNumStr.suchThat(newCol => newCol.nonEmpty && newCol =!= oldCol) // Ensure the replacement differs from original.
                    else Gen.const(oldCol) // Use the original value since this index isn't one at which to update.
                }
            } yield expectedHeader
            
            Gen.oneOf(genSubstitutions, genDeletions).toArbitrary
        }
        
        forAll { (expectedHeader: ExpectedHeader, records: NonEmptyList[Input.GoodRecord], includeIndexCol: Boolean) =>
            val (expHead, textRows) = 
                if includeIndexCol 
                then ("" :: expectedHeader, records.toList.zipWithIndex.map{ (r, i) => i.show_ :: recordToTextFields(r) })
                else (expectedHeader, records.toList.map(recordToTextFields))
            withTempDirectory{ (tempdir: os.Path) => 
                val infile = {
                    val f = tempdir / "input.csv"
                    os.write(f, (textFieldsToLine(expHead) :: textRows.map(textFieldsToLine)))
                    f
                }
                intercept[IllegalHeaderException]{ workflow(inputFile = infile, outputFolder = tempdir / "output") } match {
                    case IllegalHeaderException(header, missing) => 
                        header shouldEqual expHead
                        (AllReqdColumns.toSet -- expHead.toSet).toNonEmptySet match {
                            case None => throw new Exception(
                                s"All required columns are in expected header, leaving nothing for expected missing! $expHead"
                                )
                            case Some(expMiss) => missing shouldEqual expMiss
                        }
                }
            }
        }
    }

    test("Any row with too few fields, too many fields, or improperly typed value(s) breaks the parse.") {
        type Mutate = List[String] => List[String]
        given showForStringOrDouble: Show[String | Double] = Show.show((_: String | Double) match {
            case x: Double => x.toString
            case s: String => s
        }) // used for .show'ing a generated value of more than one type possibility
        
        def genDrops: Gen[Mutate] = Gen.atLeastOne((0 until AllReqdColumns.length)).map {
            indices => { (_: List[String]).zipWithIndex.filterNot((_, i) => indices.toSet.contains(i)).map(_._1) }
        }
        def genAdditions: Gen[Mutate] = 
            Gen.resize(5, Gen.nonEmptyListOf(Gen.choose(-3e-3, 3e3).map(_.toString))).map(additions => (_: List[String]) ++ additions)
        def genImproperlyTyped: Gen[Mutate] = {
            def genMutate[A : SimpleShow](col: String, alt: Gen[A]): Gen[Mutate] = {
                val idx = AllReqdColumns.zipWithIndex.find(_._1 === col).map(_._2).getOrElse{
                    throw new Exception(s"Cannot find index for alleged input column: $col")
                }
                alt.map(a => { (_: List[String]).updated(idx, a.show_) })
            }
            // NB: Skipping bad position and region b/c so long as they're String-ly typed, there's no way to generate a bad value.
            Gen.oneOf(Input.XCoordinateColumn, Input.YCoordinateColumn, Input.ZCoordinateColumn).flatMap(genMutate(_, Gen.alphaStr))
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
        
        forAll (genBadRecords, arbitrary[Boolean]) { case ((expBadRows, textRecords), includeIndexCol) =>
            withTempDirectory{ (tempdir: os.Path) => 
                val infile = tempdir / "input.csv"
                os.isFile(infile) shouldBe false
                val (head, textRows) = 
                    if includeIndexCol 
                    then ("" :: AllReqdColumns, textRecords.zipWithIndex.map((r, i) => i.show_ :: r))
                    else (AllReqdColumns, textRecords)
                os.write(infile, (head :: textRows.toList) map textFieldsToLine)
                os.isFile(infile) shouldBe true
                val error = intercept[Input.BadRecordsException]{ workflow(inputFile = infile, outputFolder = tempdir / "output") }
                error.records.map(_.lineNumber) shouldEqual expBadRows
            }
        }
    }

    test("Output file always has the correct header.") {
        forAll { (records: NonEmptyList[Input.GoodRecord], includeIndexCol: Boolean) => 
            withTempDirectory{ (tempdir: os.Path) => 
                val infile = tempdir / "input_with_suffix.some.random.extensions.csv"
                val (head, textRows) = 
                    if includeIndexCol
                    then ("" :: AllReqdColumns, records.zipWithIndex.map((r, i) => i.show_ :: recordToTextFields(r)))
                    else (AllReqdColumns, records.map(recordToTextFields))
                os.write(infile, (head :: textRows.toList).map(textFieldsToLine))
                val outfolder = tempdir / "output"
                val expOutfile = outfolder / "input_with_suffix.pairwise_distances__regional.csv"
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
        import io.github.iltotore.iron.autoRefine
        
        def buildPoint(x: Double, y: Double, z: Double) = Point3D(XCoordinate(x), YCoordinate(y), ZCoordinate(z))
        
        val pos = PositionName("P0001.zarr")
        val inputRecords = NonnegativeInt.indexed(List((2.0, 1.0, -1.0), (1.0, 5.0, 0.0), (3.0, 0.0, 2.0))).map{
            (pt, i) => Input.GoodRecord(pos, RegionId.unsafe(i), buildPoint.tupled(pt))
        }
        val expected: Iterable[OutputRecord] = List(0 -> 1, 0 -> 2, 1 -> 2).map{ (i, j) => 
            val nn1 = NonnegativeInt.unsafe(i)
            val nn2 = NonnegativeInt.unsafe(j)
            val rec1 = inputRecords(i)
            val rec2 = inputRecords(j)
            val reg1 = RegionId.unsafe(i)
            val reg2 = RegionId.unsafe(j)
            OutputRecord(pos, reg1, reg2, EuclideanDistance.between(rec1.point, rec2.point), nn1, nn2)
        }
        val observed = inputRecordsToOutputRecords(NonnegativeInt.indexed(inputRecords))
        observed shouldEqual expected
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

    test("Any output record's original record indices map them back to input records with identical FOV.") {
        /* To encourage collisions, narrow the choices for grouping components. */
        given arbPos: Arbitrary[PositionIndex] = Gen.oneOf(0, 1).map(PositionIndex.unsafe).toArbitrary
        forAll (Gen.choose(10, 100).flatMap(Gen.listOfN(_, arbitrary[Input.GoodRecord]))) { (records: List[Input.GoodRecord]) => 
            val indexedRecords = NonnegativeInt.indexed(records)
            val getKey = indexedRecords.map(_.swap).toMap.apply.andThen(Input.getGroupingKey)
            val observed = inputRecordsToOutputRecords(indexedRecords)
            observed.filter{ r => getKey(r.inputIndex1) === getKey(r.inputIndex2) } shouldEqual observed
        }
    }

    test("(FOV, region ID) is NOT a key!") {
        import io.github.iltotore.iron.autoRefine

        /* To encourage collisions, narrow the choices for grouping components. */
        given arbPosition: Arbitrary[PositionName] = Gen.const(PositionName("P0002.zarr")).toArbitrary
        given arbRegion: Arbitrary[RegionId] = Gen.oneOf(40, 41, 42).map(RegionId.unsafe).toArbitrary
        forAll (Gen.choose(5, 10).flatMap(Gen.listOfN(_, arbitrary[Input.GoodRecord]))) { (records: List[Input.GoodRecord]) => 
            // Pretest: must be multiple records of same region even within same FOV.
            records.groupBy(r => r.position -> r.region).view.mapValues(_.length).toMap.filter(_._2 > 1).nonEmpty shouldBe true
            val getKey = (_: Input.GoodRecord | OutputRecord) match {
                case i: Input.GoodRecord => i.position
                case o: OutputRecord => o.position
            }
            val expGroupSizes = records.groupBy(getKey).view.mapValues(g => g.size `choose` 2).toMap
            val obsGroupSizes = inputRecordsToOutputRecords(NonnegativeInt.indexed(records)).groupBy(getKey).view.mapValues(_.size).toMap
            obsGroupSizes shouldEqual expGroupSizes
        }
    }

    /** Use arbitrary instances for components to derive an an instance for the sum type. */
    given arbitraryForGoodInputRecord(using 
        arbPos: Arbitrary[PositionName], 
        arbRegion: Arbitrary[RegionId], 
        arbPoint: Arbitrary[Point3D], 
    ): Arbitrary[Input.GoodRecord] = (arbPos, arbRegion, arbPoint).mapN(Input.GoodRecord.apply)

    // nCk, i.e. number of ways to choose k indistinguishable objects from n
    extension (n: Int)
        infix private def choose(k: Int): Int = 
            require(n >= 0 && n <= 10, s"n not in [0, 10] for nCk: $n")
            require(k <= n, s"Cannot choose more items than available: $k > $n")
            factorial(n) / (factorial(k) * factorial(n - k))

    private def factorial(n: Int): Int = 
        require(n >= 0 && n <= 10, s"Factorial arg not in [0, 10]: $n")
        (1 to n).product


    /** Convert a sequence of text fields into a single line (CSV), including newline. */
    private def textFieldsToLine = Delimiter.CommaSeparator.join(_: List[String]) ++ "\n"

    /** Convert each ADT value to a simple sequence of text fields, for writing to format like CSV. */
    private def recordToTextFields = 
        given SimpleShow[Double] = SimpleShow.fromToString
        (r: Input.GoodRecord) => 
            val (x, y, z) = (r.point.x, r.point.y, r.point.z)
            List(r.position.show_, r.region.show_, x.show_, y.show_, z.show_)
end TestComputeRegionPairwiseDistances

