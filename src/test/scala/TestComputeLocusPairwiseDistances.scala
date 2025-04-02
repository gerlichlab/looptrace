package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.*
import cats.syntax.all.*
import io.github.iltotore.iron.scalacheck.all.given
import mouse.boolean.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanDistance
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.imaging.{
    FieldOfView,
    ImagingTimepoint, 
    PositionName, 
}
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.nonnegativeInt.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.* // for .show_ syntax
import at.ac.oeaw.imba.gerlich.gerlib.testing.instances.all.given

import at.ac.oeaw.imba.gerlich.looptrace.ComputeLocusPairwiseDistances.*
import at.ac.oeaw.imba.gerlich.looptrace.OneBasedFourDigitPositionName.given
import at.ac.oeaw.imba.gerlich.looptrace.collections.*
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.TraceGroupColumnName
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.ComputeLocusPairwiseDistances.Input.getGroupingKey

/**
 * Tests for the simple pairwise distances computation program, for locus-specific spots
 *
 * @author Vince Reuter
 */
class TestComputeLocusPairwiseDistances extends AnyFunSuite, ScalaCheckPropertyChecks, LooptraceSuite, should.Matchers:
    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)
    
    val AllRequiredColumns = List(
        Input.FieldOfViewColumn, 
        TraceGroupColumnName.value,
        Input.TraceIdColumn, 
        Input.LocusSpecificBarcodeTimepointColumn, 
        Input.RegionalBarcodeTimepointColumn, 
        Input.ZCoordinateColumn,
        Input.YCoordinateColumn, 
        Input.XCoordinateColumn,
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

    test("Input file that's just a header produces an output file that's empty.") {
        withTempDirectory{ (tempdir: os.Path) => 
            /* Setup and pretests */
            val infile = tempdir / "input.csv"
            val outfolder = tempdir / "output"
            os.makeDir(outfolder)
            os.write(infile, Delimiter.CommaSeparator.join(AllRequiredColumns) ++ "\n")
            val expOutfile = outfolder / "input.pairwise_distances__locus_specific.csv"
            os.exists(expOutfile) shouldBe false
            workflow(inputFile = infile, outputFolder = outfolder)
            os.isFile(expOutfile) shouldBe true
            safeReadAllWithOrderedHeaders(expOutfile) match {
                case Left(err) => fail(s"Expected successful output file parse but got error: $err")
                case Right((header, Nil)) => header shouldEqual List()
                case Right((_, records)) => fail(s"Expected empty output file but got ${records.length} record(s)!")

            }
        }
    }

    test("Trying to use a file with just records and no header fails the parse as expected.") {
        forAll { (records: NonEmptyList[Input.GoodRecord]) => 
            withTempDirectory{ (tempdir: os.Path) => 
                val infile = tempdir / "input.csv"
                os.write(infile, records.toList.map(recordToTextFields `andThen` rowToLine))
                val expError = {
                    val textHead = recordToTextFields(records.head)
                    // Account for the fact that randomly drawn first-row elements could collide with 
                    // a required header field and therefore reduce the theoretically missing set.
                    val expMiss = (AllRequiredColumns.toNel.get.toNes -- textHead.toNel.get.toNes).toNonEmptySetUnsafe
                    IllegalHeaderException(textHead, expMiss)
                }
                val obsError = intercept[IllegalHeaderException]{ workflow(inputFile = infile, outputFolder = tempdir / "output") }
                obsError shouldEqual expError
            }
        }
    }

    test("Any nonempty subset of missing/incorrect columns from input file causes expected error.") {
        type ExpectedHeader = List[String]
        given [A] => Shrink[A] = Shrink.shrinkAny[A]
        
        def genExpHeadAndMiss: Gen[(ExpectedHeader, NonEmptySet[String])] = {
            def genDeletions: Gen[(ExpectedHeader, NonEmptySet[String])] = Gen.choose(1, AllRequiredColumns.length - 1)
                .flatMap(Gen.pick(_, (0 until AllRequiredColumns.length)))
                .map{ indices => 
                    val (expHead, expMiss) = AllRequiredColumns.zipWithIndex.partition((_, i) => !indices.contains(i))
                    expHead.map(_._1) -> expMiss.map(_._1).toNel.get.toNes
                }
            def genSubstitutions: Gen[(ExpectedHeader, NonEmptySet[String])] = for
                indicesToChange <- Gen.atLeastOne((0 until AllRequiredColumns.length)).map(_.toSet) // Choose header fields to change.
                expHead <- AllRequiredColumns.zipWithIndex.traverse{ (col, idx) => 
                    if indicesToChange contains idx 
                    then Gen.alphaNumStr.suchThat(_ =!= col) // Ensure the replacement differs from original.
                    else Gen.const(col) // Use the original value since this index isn't one at which to update.
                }
            yield (expHead, indicesToChange.toList.toNel.get.toNes.map(AllRequiredColumns.zipWithIndex.map(_.swap).toMap.apply))            
            Gen.oneOf(genSubstitutions, genDeletions).suchThat{ (head, miss) => (head.toSet & miss.toSortedSet).isEmpty }
        }
        
        forAll (genExpHeadAndMiss, arbitrary[NonEmptyList[Input.GoodRecord]]) { case ((expectedHead, expectedMiss), records) =>
            withTempDirectory{ (tempdir: os.Path) => 
                val infile = {
                    val f = tempdir / "input.csv"
                    os.write(f, (expectedHead :: records.toList.map(recordToTextFields)).map(rowToLine))
                    f
                }
                intercept[IllegalHeaderException]{ workflow(inputFile = infile, outputFolder = tempdir / "output") } match {
                    case IllegalHeaderException(observedHead, observedMiss) => 
                        observedHead shouldEqual expectedHead
                        observedMiss shouldEqual expectedMiss

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
        
        def genDrops: Gen[Mutate] = Gen.atLeastOne((0 until AllRequiredColumns.length)).map {
            indices => { (_: List[String]).zipWithIndex.filterNot((_, i) => indices.toSet.contains(i)).map(_._1) }
        }
        def genAdditions: Gen[Mutate] = 
            Gen.resize(5, Gen.nonEmptyListOf(Gen.choose(-3e-3, 3e3).map(_.toString))).map(additions => (_: List[String]) ++ additions)
        def genImproperlyTyped: Gen[Mutate] = {
            def genMutate[A : Show](col: String, alt: Gen[A]): Gen[Mutate] = {
                val idx = AllRequiredColumns.zipWithIndex.find(_._1 === col).map(_._2).getOrElse{
                    throw new Exception(s"Cannot find index for alleged input column: $col")
                }
                alt.map(a => { (_: List[String]).updated(idx, a.show) })
            }
            def genBadPosition: Gen[Mutate] = 
                given Show[Int | Double] = Show.show(_.toString)
                // PositionName (for field of view) must not be numeric.
                genMutate(Input.FieldOfViewColumn, Gen.oneOf(arbitrary[Int], arbitrary[Double]))
            def genBadTrace: Gen[Mutate] = genMutate(Input.TraceIdColumn, Gen.oneOf(Gen.alphaStr, arbitrary[Double]))
            // NB: Skipping bad region b/c so long as it's String-ly typed, there's no way to generate a bad value.
            def genBadLocus: Gen[Mutate] = genMutate(Input.LocusSpecificBarcodeTimepointColumn, Gen.oneOf(Gen.alphaStr, arbitrary[Double]))
            def genBadPoint: Gen[Mutate] = 
                Gen.oneOf(Input.XCoordinateColumn, Input.YCoordinateColumn, Input.ZCoordinateColumn).flatMap(genMutate(_, Gen.alphaStr))
            Gen.oneOf(genBadPoint, genBadLocus, genBadTrace, genBadPosition)
        }
        
        def genBadRecords: Gen[(NonEmptyList[Int], NonEmptyList[List[String]])] = for
            goods <- arbitrary[NonEmptyList[Input.GoodRecord]].map(_.map(recordToTextFields))
            indices <- Gen.atLeastOne((0 until goods.length)).map(_.toList.sorted.toNel.get)
            mutations <- indices.toList
                // Put genImproperlyTyped first since there are more possbilities there, 
                // and the distribution is biased towards selection of elements earlier in the sequence.
                .traverse{ i => Gen.oneOf(genImproperlyTyped, genAdditions, genDrops).map(i -> _) }
                .map(_.toMap)
        yield (indices, goods.zipWithIndex.map{ (r, i) => mutations.get(i).fold(r)(_(r)) })
        
        forAll (genBadRecords) { (expBadRows, textRecords) =>
            withTempDirectory{ (tempdir: os.Path) => 
                val infile = tempdir / "input.csv"
                os.isFile(infile) shouldBe false
                writeMinimalInputCsv(infile, AllRequiredColumns :: textRecords.toList)
                os.isFile(infile) shouldBe true
                val error = intercept[Input.BadRecordsException]{ workflow(inputFile = infile, outputFolder = tempdir / "output") }
                val obsBadRows = error.records.map(_.lineNumber)
                val missingBadRows: Set[Int] = expBadRows.toList.toSet -- obsBadRows.toList.toSet
                val extraBadRows: Set[Int] = obsBadRows.toList.toSet -- expBadRows.toList.toSet
                if missingBadRows.nonEmpty
                then fail(s"${missingBadRows.size} row(s) expected bad but not: ${textRecords.zipWithIndex.filter((_, i) => missingBadRows.contains(i))}")
                else if extraBadRows.nonEmpty
                then fail(s"${extraBadRows.size} row(s) unexpectedly bad: ${textRecords.zipWithIndex.filter((_, i) => extraBadRows.contains(i))}")
                else succeed
            }
        }
    }

    test("Distances computed are accurately Euclidean.") {
        def buildPoint(x: Double, y: Double, z: Double) = Point3D(XCoordinate(x), YCoordinate(y), ZCoordinate(z))
        val pos = OneBasedFourDigitPositionName.fromString(false)("P0001").fold(msg => throw new Exception(msg), identity)
        val traceGroup = TraceGroupMaybe.empty
        val tid = TraceId(NonnegativeInt(1))
        val reg = RegionId(ImagingTimepoint(NonnegativeInt(40)))
        val inputRecords = NonnegativeInt.indexed(List((2.0, 1.0, -1.0), (1.0, 5.0, 0.0), (3.0, 0.0, 2.0))).map{
            (pt, i) => Input.GoodRecord(pos, traceGroup, tid, reg, LocusId(ImagingTimepoint(i)), buildPoint.tupled(pt))
        }
        val getExpEuclDist = (i: Int, j: Int) => EuclideanDistance.between(inputRecords(i).point, inputRecords(j).point)
        def rawAndTime: Int => (NonnegativeInt, LocusId) = n => (n, n).bimap(NonnegativeInt.unsafe, LocusId.unsafe)
        val expected: Iterable[OutputRecord] = List(0 -> 1, 0 -> 2, 1 -> 2).map{ (i, j) => 
            val (idx1, loc1) = rawAndTime(i)
            val (idx2, loc2) = rawAndTime(j)
            OutputRecord(pos, traceGroup, tid, reg, reg, loc1, loc2, getExpEuclDist(i, j), idx1, idx2)
        }
        val observed = inputRecordsToOutputRecords(NonnegativeInt.indexed(inputRecords))
        observed shouldEqual expected
    }

    test("Trace ID is unique globally for an experiment: records with the same trace ID but different fields of view (FOVs) causes expected error. Issue #390"):
        import at.ac.oeaw.imba.gerlich.gerlib.syntax.tuple2.*
        
        val posNames = List("P0001.zarr", "P0002.zarr").map(PositionName.unsafe)
        
        given Arbitrary[(PositionName, TraceId)] = 
            // Restrict to relatively few choices of trace ID and position name, to boost collision probability.
            Gen.oneOf(posNames.map(_ -> TraceId.unsafe(8))).toArbitrary

        forAll (Gen.resize(10, arbitrary[Inputs]).flatMap(makeTraceUniqueInLoci)) { records => 
            val possiblePrefixes = records.groupBy(getGroupingKey) // NB: This grouping SHOULD be irrelevant since the trace ID is fixed to a constant value.
                .view
                .values
                .flatMap(_.combinations(2).toList.flatMap{ // Check that at least one pair is a case of same trace ID but with different FOv.
                    case r1 :: r2 :: Nil => 
                        (r1.trace === r2.trace && r1.fieldOfView =!= r2.fieldOfView).option{
                            val (fov1, fov2) = (r1, r2).flipBy(_.locus).mapBoth(_.fieldOfView) // The records will have been ordered such that locus1 <= locus2.
                            s"Different FOVs ($fov1 and $fov2) for records from the same trace (${r1.trace})"
                        }
                    case rs => throw new Exception(s"Got ${rs.length} records when taking pairs!")
                })
                .toList
            whenever(possiblePrefixes.nonEmpty):
                val obsMsg = intercept[Exception]{ inputRecordsToOutputRecords(NonnegativeInt.indexed(records)) }.getMessage
                if possiblePrefixes.exists(obsMsg.startsWith) then succeed
                else
                    //for (pp <- possiblePrefixes) { println(pp) }; println(obsMsg) 
                    fail(s"Observed error message: $obsMsg. Possibles: $possiblePrefixes")
        }

    test("Trace ID is the grouping key; records with different region ID but the same trace ID still have distance between them computed. Issue #390"):
        val numCombos2 = (n: Int) => n * (n - 1) / 2
        // Generate records such that there's at least one which has a different region ID, but all have the same trace ID.
        // This ensures that the number of output records differs w.r.t. whether or not the computation does or does not pair up records with the same trace ID but different region ID.
        given Arbitrary[TraceId] = Gen.const(TraceId.unsafe(10)).toArbitrary
        given Arbitrary[PositionName] = Gen.const(PositionName.unsafe("P0001.zarr")).toArbitrary

        def genInputs: Gen[Inputs] = 
            // Here, use a small input size so that the search for unique locus IDs doesn't exhause the max failure count for Gen.setOfN.
            Gen.resize(10, arbitrary[Inputs].flatMap(makeTraceUniqueInLoci))

        forAll (genInputs) { inputRecords => 
            val outputRecords = inputRecordsToOutputRecords(NonnegativeInt.indexed(inputRecords))
            val expNumDiscard: Int = // Distance won't be computed between records with the same locus ID within a trace ID.
                inputRecords.groupBy(_.locus)
                    .view
                    .mapValues(_.size)
                    .values
                    .map(numCombos2)
                    .sum
            outputRecords.size shouldEqual numCombos2(inputRecords.size) - expNumDiscard
        }

    test("Unexpected header error can't be created with the empty missing columns or with an unnecessary column.") {
        assertTypeError{ "IllegalHeaderException(List(), List())" }
        assertCompiles{ "IllegalHeaderException(List(), NonEmptySet.one(\"x\"))" }
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
        given arbPosAndTrace: Arbitrary[(PositionName, TraceId)] = genPosTracePairOneOrOther.toArbitrary
        given arbRegion: Arbitrary[RegionId] = Gen.oneOf(40, 41).map(RegionId.unsafe).toArbitrary

        given Arbitrary[Inputs] = getArbitraryForGoodInputRecords(RequestForArbitraryInputs.repeatsForbidden)

        forAll { (records: List[Input.GoodRecord]) => 
            val indexedRecords = NonnegativeInt.indexed(records)
            val getKey = indexedRecords.map(_.swap).toMap.apply.andThen(Input.getGroupingKey)
            val observed = inputRecordsToOutputRecords(indexedRecords)
            observed.filter{ r => getKey(r.inputIndex1) === getKey(r.inputIndex2) } shouldEqual observed
        }
    }

    test("Any trace ID with more than one instance of a particular locus ID causes the expected error.") {
        /* To encourage collisions, narrow the choices for grouping components. */
        given arbPosAndTrace: Arbitrary[(PositionName, TraceId)] = genPosTracePairOneOrOther.toArbitrary
        given arbRegion: Arbitrary[RegionId] = Gen.oneOf(40, 41).map(RegionId.unsafe).toArbitrary
        given arbTime: Arbitrary[ImagingTimepoint] = Gen.const(ImagingTimepoint(NonnegativeInt(10))).toArbitrary
        
        given Arbitrary[Inputs] = getArbitraryForGoodInputRecords(RequestForArbitraryInputs.RepeatsAllowed)

        forAll { (records: List[Input.GoodRecord]) => 
            whenever(records
                .groupBy(_.trace)
                .values
                .exists(trace => !recordsSatisfyUniqueLocusPerTraceConstraint(trace))
            ):
                val error = intercept[Exception]{ inputRecordsToOutputRecords(NonnegativeInt.indexed(records)).toList }
                error.getMessage.contains("grouped for locus pairwise distance computation have the same locus ID") shouldBe true
        }
    }

    test("Record count / number of records is as expected -- locus ID pairs are unordered (equivalent when reflected). Issue #395"):
        // For this test, fix the randomisation to use a single field of view and a single trace ID.
        given Arbitrary[(PositionName, TraceId)] = Gen.const(PositionName.unsafe("P0001.zarr") -> TraceId.unsafe(0)).toArbitrary
        given Arbitrary[Inputs] = arbitrary[Inputs].flatMap(makeTraceUniqueInLoci).toArbitrary

        forAll (Gen.resize(15, arbitrary[Inputs])) { inputs =>  
            val n = inputs.length
            val numExp = n * (n - 1) / 2
            val outputs = inputRecordsToOutputRecords(NonnegativeInt.indexed(inputs))
            val numObs = outputs.size
            numObs shouldEqual numExp
        }

    test("The pairs of locus IDs are as expected; namely, each record has the pair of locus IDs ordered with the lesser one first, and within a trace, these pairs are properly sorted."):
        // For this test, fix the randomisation to use a single field of view and a single trace ID.
        given Arbitrary[(PositionName, TraceId)] = Gen.const(PositionName.unsafe("P0001.zarr") -> TraceId.unsafe(0)).toArbitrary
        given Arbitrary[Inputs] = arbitrary[Inputs].flatMap(makeTraceUniqueInLoci).toArbitrary

        forAll (Gen.resize(15, arbitrary[Inputs])) { inputs => 
            whenever(inputs.length > 1): // Single or empty will generate empty output
                /* Validate that the restriction to a single FOV and a single trace ID worked. */
                inputs.map(_.fieldOfView).toSet.size shouldEqual 1 // This should be handled by the construction of the Arbitrary[(PosName, TraceId)] instance.
                inputs.map(_.trace).toSet.size shouldEqual 1 // This should be handled by the construction of the Arbitrary[(PosName, TraceId)] instance.
                inputs.map(_.traceGroup).toSet.size shouldEqual 1 // This should be handled by setting 

                /* Now formulate and verify the actual expectation about the locus IDs. */
                val locusIds = inputs.map(_.locus)
                val expected = locusIds.combinations(2).map(_.sorted(Order[LocusId].toOrdering))
                    .flatMap{
                        case i1 :: i2 :: Nil => (i1, i2).some
                        case ids => throw new Exception(s"Got ${ids.length} IDs when taking pairs of 2!")
                    }
                    .toList
                    .sorted(Order[(LocusId, LocusId)].toOrdering)
                val observed = inputRecordsToOutputRecords(NonnegativeInt.indexed(inputs))
                    .toList
                    .map(r => r.locus1 -> r.locus2)
                expected shouldEqual observed
        }

    /** Treat trace ID generation equivalently to ROI index generation. */
    given (arbRoiIdx: Arbitrary[RoiIndex]) => Arbitrary[TraceId] = arbRoiIdx.map(TraceId.fromRoiIndex)
    
    given (
        arbPosAndTrace: Arbitrary[(OneBasedFourDigitPositionName, TraceId)],
        arbRegion: Arbitrary[RegionId], 
        arbLocus: Arbitrary[LocusId], 
        arbPoint: Arbitrary[Point3D], 
    ) => Arbitrary[Input.GoodRecord] = 
        (arbPosAndTrace, arbRegion, arbLocus, arbPoint)
            .mapN{ case ((pos, trace), reg, loc, pt) => 
                Input.GoodRecord(pos, TraceGroupMaybe.empty, trace, reg, loc, pt) 
            }

    private type Inputs = List[Input.GoodRecord]

    enum RequestForArbitraryInputs:
        case RepeatsForbidden(maxFailures: Int = 50) extends RequestForArbitraryInputs
        case RepeatsAllowed extends RequestForArbitraryInputs

    object RequestForArbitraryInputs:
        def repeatsForbidden = RequestForArbitraryInputs.RepeatsForbidden()

    def getArbitraryForGoodInputRecords(maxFailures: Int): Arbitrary[Inputs] = 
        arbitrary[Inputs].flatMap{ records => 
            val gens: List[Gen[Inputs]] = records
                .groupBy{ (r: Input.GoodRecord) => r.fieldOfView -> r.trace }
                .view
                .values
                .toList
                .foldRight(List.empty[Gen[Inputs]]){(traceGroup, acc) => 
                    val replacement = 
                        if recordsSatisfyUniqueLocusPerTraceConstraint(traceGroup)
                        then Gen.const(traceGroup)
                        else makeTraceUniqueInLoci(traceGroup, maxFailures)
                    replacement :: acc
                }
            gens.sequence
        }
        .map(_.foldRight(List.empty[Input.GoodRecord]){ (traceGroup, acc) => traceGroup ::: acc })
        .toArbitrary

    private def makeTraceUniqueInLoci(records: Inputs): Gen[Inputs] = makeTraceUniqueInLoci(records, 50)

    private def makeTraceUniqueInLoci(records: Inputs, maxFailures: Int)(using Arbitrary[LocusId]): Gen[Inputs] = 
        import org.scalacheck.ops.* // for Gen.setOfN
        Gen.setOfN(records.length, maxFailures, arbitrary[LocusId])
            .map(_.toList.zip(records).map{ (loc, rec) => rec.copy(locus = loc) })

    private def getArbitraryForGoodInputRecords(
        request: RequestForArbitraryInputs = RequestForArbitraryInputs.repeatsForbidden
    )(using Arbitrary[Input.GoodRecord]): Arbitrary[Inputs] = 
        request match {
            case RequestForArbitraryInputs.RepeatsAllowed => summon[Arbitrary[Inputs]]
            case RequestForArbitraryInputs.RepeatsForbidden(maxFailures) => getArbitraryForGoodInputRecords(maxFailures)
        }

    private def recordsSatisfyUniqueLocusPerTraceConstraint(records: List[Input.GoodRecord]): Boolean = 
        records.combinations(2).forall{
            case r1 :: r2 :: Nil => r1.locus =!= r2.locus
            case rs => throw new Exception(s"Got ${rs.length} records when taking combinations of 2!")
        }

    private def genPosTracePairOneOrOther: Gen[(PositionName, TraceId)] = 
        val posNames = List("P0001.zarr", "P0002.zarr") map PositionName.unsafe
        val traceIds = List(2, 3) map TraceId.unsafe
        Gen.oneOf(posNames zip traceIds)

    /** Write the given rows as CSV to the given file. */
    private def writeMinimalInputCsv(f: os.Path, rows: List[List[String]]): Unit = 
        os.write(f, rows map rowToLine)

    /** Write the given records as CSV to the given file. */
    private def writeMinimalInputCsv(f: os.Path, records: NonEmptyList[Input.GoodRecord]): Unit = 
        writeMinimalInputCsv(f, AllRequiredColumns :: records.toList.map(recordToTextFields))

    /** Convert a sequence of text fields into a single line (CSV), including newline. */
    private def rowToLine = Delimiter.CommaSeparator.join(_: List[String]) ++ "\n"

    /** Convert each ADT value to a simple sequence of text fields, for writing to format like CSV. */
    private def recordToTextFields = 
        import at.ac.oeaw.imba.gerlich.gerlib.instances.simpleShow.given // for SimpleShow[String]
        import at.ac.oeaw.imba.gerlich.looptrace.instances.traceId.given // for SimpleShow[TraceGroupMaybe]
        given SimpleShow[Double] = SimpleShow.fromToString
        (r: Input.GoodRecord) => 
            val (x, y, z) = (r.point.x, r.point.y, r.point.z)
            List(r.fieldOfView.show_, r.traceGroup.show_, r.trace.show_, r.region.show_, r.locus.show_, x.show_, y.show_, z.show_)
end TestComputeLocusPairwiseDistances
