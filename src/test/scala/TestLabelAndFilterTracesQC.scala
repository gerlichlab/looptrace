package at.ac.oeaw.imba.gerlich.looptrace

import scala.io.Source
import org.scalacheck.{ Gen, Shrink }
import org.scalactic.Equality
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.funsuite.AnyFunSuite

import LabelAndFilterTracesQC.{
    DistanceToRegion, 
    ParserConfig, 
    QcPassColumn, 
    QCResult, 
    SigmaXY, 
    SigmaZ, 
    SignalToNoise, 
    workflow
}
import PathHelpers.*

/** Tests for the filtration of the individual supports (single FISH probes) of chromatin fiber traces */
class TestLabelAndFilterTracesQC extends AnyFunSuite, GenericSuite, ScalacheckSuite, should.Matchers:
    
    test("Collision between any of the column name values in a parser config is prohibited.") { pending }
    
    test("Non-default config can work if appropriate column names are found.") { pending }

    test("Absence of any required column fails the parse.") { pending }

    test("Any row with inappropriate field count fails the parse expectedly, as it suggests a corrupted data file.") { pending }

    test("Absence of header fails the parser expectedly.") { pending }

    test("Inability to infer the delimiter fails the parse expectedly.") { pending }

    test("Python nulls (numeric or otherwise) are handled appropriately.") { pending }

    test("Missing distance-to-region column, specifically, fails the parse expectedly.") { pending }

    test("Missing frame names column, specifically, fails the parse expectedly.") { pending }

    test("Basic golden path test") {
        withTempDirectory{ (tempdir: os.Path) => 

            /* Pretest: equivalence between expected columns and concatenation of input columns with QC component fields */
            val expLinesUnfiltered = os.read.lines(componentExpectationFile)
            val expColumnsUnfiltered = {
                val sep = Delimiter.fromPathUnsafe(componentExpectationFile)
                sep `split` expLinesUnfiltered.head
            }.toList
            val componentLabelColumns: List[String] = labelsOf[QCResult].productIterator.toList.map(_.asInstanceOf[String])
            val inputHeaderFields = {
                val headline = os.read.lines(tracesInputFile).head
                val sep = Delimiter.fromPathUnsafe(componentExpectationFile)
                sep `split` headline
            }.toList
            val qcColumns = componentLabelColumns :+ QcPassColumn
            expColumnsUnfiltered shouldEqual {
                (if inputHeaderFields.head === "" then inputHeaderFields.tail else inputHeaderFields) ++ qcColumns
            }

            // Run all tests over both input files -- with or without index_col=0 -- as results should be invariant.
            forAll (Table("infile", tracesInputFile, tracesInputFileWithoutIndex)) { infile => 
                val (expUnfilteredPath, expFilteredPath) = pretest(tempdir = tempdir, infile = infile)

                // With the pretest passed, run the action that generate outputs from inputs.
                runStandardWorkflow(infile = infile, outfolder = tempdir)

                /* Now, first, just do basic existence check of files... */
                List(infile, expUnfilteredPath, expFilteredPath).map(os.isFile) shouldEqual List(true, true, true)

                val obsLinesUnfiltered = os.read.lines(expUnfilteredPath)

                // ...then, check that no probes were excluded...
                obsLinesUnfiltered.length shouldEqual expLinesUnfiltered.length

                // ...now, use the observed and expected rows to do more sophisticated checks...
                withCsvPair(expUnfilteredPath, componentExpectationFile){ (obsRows: CsvRows, expRows: CsvRows) =>
                    // The number of observed output rows should be the 1 less (header) than expected lines count.
                    obsRows.size shouldEqual expLinesUnfiltered.length - 1

                    // Each row should have the expected fields (column names from header).
                    forAll (Table("fields", obsRows.toList*)) { row => 
                        /* Observed row field names should equal expected row field names. */
                        (row.keySet -- expColumnsUnfiltered, expColumnsUnfiltered.filterNot(row.keySet.contains)) shouldEqual (Set(), List())
                    }

                    // Every QC result (component and aggregate) column should be as expected.
                    forAll (Table("col", qcColumns*)) { col =>
                        val obs = obsRows.map(_(col))
                        val exp = expRows.map(_(col))
                        obs shouldEqual exp
                    }

                    // Finally, do a stricter line-by-line equality test, for the UNFILTERED output.
                    assertPairwiseEquality(observed = obsLinesUnfiltered.toList, expected = expLinesUnfiltered.toList)

                    // Repeat the line-by-line equality test for the FILTERED output.
                    val obsLinesFiltered = os.read.lines(expFilteredPath)
                    val expLinesFiltered = os.read.lines(wholemealFilteredExpectationFile)
                    assertPairwiseEquality(observed = obsLinesFiltered.toList, expected = expLinesFiltered.toList)
                }
            }
            
        }
    }

    test("Probe exclusion is correct.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        // probe names taken from the frame_name column of the traces input file
        val probeNames = Set("pre_image", "Dp027", "Dp028", "Dp027_repeat", "Dp105", "Dp107", "blank_1", "Dp019", "Dp020", "Dp021")
        forAll (Gen.zip(Gen.someOf(probeNames), Gen.oneOf(tracesInputFile, tracesInputFileWithoutIndex))) {
            case (exclusions, infile) => 
                withCsvData(infile){ (rows: CsvRows) => rows.map(_(ParserConfig.default.frameNameColumn)).toSet shouldEqual probeNames }
                withTempDirectory{ (tempdir: os.Path) => 
                    // Perform the pretest and get the expected result paths.
                    val (expUnfilteredPath, expFilteredPath) = pretest(tempdir = tempdir, infile = infile)
                    // Run the output-generating action under test.
                    runStandardWorkflow(infile = infile, outfolder = tempdir, probeExclusions = exclusions.map(ProbeName.apply).toList)
                    // Now, first, just do basic existence check of files...
                    List(infile, expUnfilteredPath, expFilteredPath).map(os.isFile) shouldEqual List(true, true, true)

                    if (exclusions === probeNames) {
                        /* Each file should be just a header. */
                        withCsvData(expUnfilteredPath)((_: CsvRows).isEmpty shouldBe true)
                        withCsvData(expFilteredPath)((_: CsvRows).isEmpty shouldBe true)
                        os.read.lines(expUnfilteredPath).length shouldEqual 1
                        os.read.lines(expFilteredPath).length shouldEqual 1
                    } else {
                        /* Each file should exhibit row-by-row equality with expectation. */
                        withCsvPair(expUnfilteredPath, componentExpectationFile){ (obsRows: CsvRows, expRowsAll: CsvRows) =>
                            val expRows = expRowsAll.filter{ r => !exclusions.contains(r("frame_name")) }
                            assertPairwiseEquality(observed = obsRows.toList, expected = expRows.toList)
                        }
                        withCsvPair(expFilteredPath, wholemealFilteredExpectationFile){ (obsRows: CsvRows, expRowsAll: CsvRows) =>
                            val expRowsFilt = expRowsAll.filter{ r => !exclusions.contains(r("frame_name")) }
                            assertPairwiseEquality(observed = obsRows.toList, expected = expRowsFilt.toList)
                        }
                    }
                }
            }
    }

    test("Trace length filter is effective.") {
        implicit def noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        val allLinesFilteredExpectation = os.read.lines(wholemealFilteredExpectationFile).toList
        val minLenDemarcation = 3
        val expWhenLessThanDemarcation = allLinesFilteredExpectation
        // Take the rows grouped together by (pos ID, region ID, roi ID) where group size is at least the demarcation size.
        // Drop the first 5 lines, which correspond to 1 for the header and 4 records before the block of 3 (pid=0, rid=23, tid=1).
        val expWhenEqualToDemarcation = allLinesFilteredExpectation.head :: allLinesFilteredExpectation.drop(5).take(minLenDemarcation)
        val expWhenMoreThanDemarcation = allLinesFilteredExpectation.head :: List()
        def genMinLenAndExp = Gen.oneOf(
            Gen.choose(0, 2).map(NonnegativeInt.unsafe).map(_ -> expWhenLessThanDemarcation),
            Gen.const(NonnegativeInt(3)).map(_ -> expWhenEqualToDemarcation), 
            Gen.choose(4, Int.MaxValue).map(NonnegativeInt.unsafe).map(_ -> expWhenMoreThanDemarcation)
            )
        forAll (Gen.zip(genMinLenAndExp, Gen.oneOf(tracesInputFile, tracesInputFileWithoutIndex))) {
            case ((minTraceLen, expLinesFiltered), infile) => 
                withTempDirectory{ (tempdir: os.Path) =>
                    val (expUnfilteredPath, expFilteredPath) = pretest(tempdir = tempdir, infile = infile)
                    // With the pretest passed, run the action that generate outputs from inputs.
                    runStandardWorkflow(infile = infile, outfolder = tempdir, minTraceLength = minTraceLen)
                    // Now, first, just do basic existence check of files...
                    List(infile, expUnfilteredPath, expFilteredPath).map(os.isFile) shouldEqual List(true, true, true)

                    /* Do a strict line-by-line equality test, for the UNFILTERED output; trace length filter has no effect. */
                    assertPairwiseEquality(
                        observed = os.read.lines(expUnfilteredPath).toList, 
                        expected = os.read.lines(componentExpectationFile).toList
                        )
                    
                    /* Do a check of the lines of the filtered output file. */
                    assertPairwiseEquality(observed = os.read.lines(expFilteredPath).toList, expected = expLinesFiltered)
                }
        }
    }

    /* Ancillary functions and types */
    type CsvRows = Iterable[CsvRow]
    private def componentExpectationFile = getResourcePath("traces.labeled.unfiltered.csv")
    private def wholemealFilteredExpectationFile = getResourcePath("traces.labeled.filtered.csv")
    private def getResourcePath(name: String): os.Path = 
        os.Path(getClass.getResource(s"/TestLabelAndFilterTracesQC/$name").getPath)
    
    private def pretest(tempdir: os.Path, infile: os.Path) = {
        val expUnfilteredPath = tempdir / s"${infile.baseName}.unfiltered.csv"
        val expFilteredPath = tempdir / s"${infile.baseName}.filtered.csv"
        os.isFile(tracesInputFile) shouldBe true // Input file exists.
        os.isFile(expUnfilteredPath) shouldBe false // Unfiltered output doesn't yet exist.
        os.isFile(expFilteredPath) shouldBe false // Filtered output doesn't yet exist.
        (expUnfilteredPath, expFilteredPath)
    }
    private def tracesInputFile = getResourcePath("traces__with_index.raw.csv")
    private def tracesInputFileWithoutIndex = getResourcePath("traces__without_index.raw.csv")

    def assertPairwiseEquality[T : Equality](observed: List[T], expected: List[T]) = {
        observed.length shouldEqual expected.length
        forAll (Table(("obs", "exp"), observed.zip(expected)*)) { case (obs, exp) => obs shouldEqual exp }
    }

    private def runStandardWorkflow(
        infile: os.Path = tracesInputFile,
        outfolder: os.Path, 
        config: ParserConfig = ParserConfig.default, 
        maxDistFromRegion: DistanceToRegion = DistanceToRegion(NonnegativeReal(800)), 
        minSignalToNoise: SignalToNoise = SignalToNoise(PositiveReal(2)), 
        maxSigmaXY: SigmaXY = SigmaXY(PositiveReal(150)), 
        maxSigmaZ: SigmaZ = SigmaZ(PositiveReal(400)),
        probeExclusions: List[ProbeName] = List(), 
        minTraceLength: NonnegativeInt = NonnegativeInt(0)
        ) = workflow(config, infile, maxDistFromRegion, minSignalToNoise, maxSigmaXY, maxSigmaZ, probeExclusions, minTraceLength, outfolder)

end TestLabelAndFilterTracesQC