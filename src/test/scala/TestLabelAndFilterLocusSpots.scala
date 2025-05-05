package at.ac.oeaw.imba.gerlich.looptrace

import scala.io.Source
import cats.data.{EitherNel, NonEmptyList}
import cats.syntax.all.*
import org.scalacheck.{Gen, Shrink}
import org.scalactic.Equality
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import com.github.tototoshi.csv.*

import at.ac.oeaw.imba.gerlich.gerlib.imaging.{ImagingTimepoint, PositionName}
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

import at.ac.oeaw.imba.gerlich.looptrace.LabelAndFilterLocusSpots.{
  ParserConfig,
  QcPassColumn,
  workflow
}
import at.ac.oeaw.imba.gerlich.looptrace.LocusSpotQC.*
import at.ac.oeaw.imba.gerlich.looptrace.PathHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*
import org.scalatest.prop.TableFor2

/** Tests for the filtration of the individual supports (single FISH probes) of
  * chromatin fiber traces
  */
class TestLabelAndFilterLocusSpots
    extends AnyFunSuite,
      ScalaCheckPropertyChecks,
      GenericSuite,
      should.Matchers:
  override implicit val generatorDrivenConfig: PropertyCheckConfiguration =
    PropertyCheckConfiguration(minSuccessful = 100)

  test(
    "Collision between any of the column name values in a parser config is prohibited."
  ) { pending }

  test("Non-default config can work if appropriate column names are found.") {
    pending
  }

  test("Absence of any required column fails the parse.") { pending }

  test(
    "Any row with inappropriate field count fails the parse expectedly, as it suggests a corrupted data file."
  ) { pending }

  test("Absence of header fails the parser expectedly.") { pending }

  test("Inability to infer the delimiter fails the parse expectedly.") {
    pending
  }

  test("Python nulls (numeric or otherwise) are handled appropriately.") {
    pending
  }

  test(
    "Missing distance-to-region column, specifically, fails the parse expectedly."
  ) { pending }

  test(
    "Missing probe names column, specifically, fails the parse expectedly."
  ) { pending }

  test(
    "A spot can only be retained if the distance between the centroid of its Gaussian fit and each edge is at least the standard deviation of the spot's Gaussian fit. #265"
  ) {
    pending
  }

  test(
    "The two points files--1 for QC pass and 1 for QC fail--are always produced for each field of view, even if empty."
  ) { pending }

  test(
    "The combined number of non-header records in the QC pass and QC fail files is the number of non-header records in the unfiltered, labeled file which are displayable."
  ) { pending }

  test(
    "Counts of QC pass and fail records in their files correspond to summing over the qcPass column from the unfiltered file, and deducting non-displayable records."
  ) { pending }

  test(
    "Each points file has a header that works for napari (index, axis-0, axis-1, axis-2, axis-3, axis-4)."
  ) { pending }

  test(
    "The seequence of axis-0 values ('traceId') in each points file has endpoints 0 and T, where T is the number of unique trace IDs for the field of view."
  ) { pending }

  test(
    "The seequence of axis-0 values ('traceId') in each points file has a 'hole' exactly where there's a trace ID with no displayable points."
  ) { pending }

  test(
    "Each points file is correctly sorted in ascending order of (traceId, timepont), i.e. ('axis-0', 'axis-1') for napar."
  ) { pending }

  test(
    "Filter for maximal distance between a locus spot center and its from region spot center works."
  ) { pending }

  test("Filter for minimum signal-to-noise ratio (SNR) works.") { pending }

  test("Filter for concentration in XY works (maxSigmaXY is respected.)") {
    pending
  }

  test("Filter for concentration in Z works (maxSigmaZ is respected.)") {
    pending
  }

  test("Filter for within-1-stanard-deviation-of-bounds is correct in Z.") {
    pending
  }

  test("Filter for within-1-stanard-deviation-of-bounds is correct in Y.") {
    pending
  }

  test("Filter for within-1-stanard-deviation-of-bounds is correct in X.") {
    pending
  }

  test("Basic golden path test") {

    /* Pretest: equivalence between expected columns and concatenation of input columns with QC component fields */
    val expLinesUnfiltered = os.read.lines(componentExpectationFile)
    val expColumnsUnfiltered = {
      val sep = Delimiter.fromPathUnsafe(componentExpectationFile)
      sep `split` expLinesUnfiltered.head
    }.toList
    val componentLabelColumns: List[String] =
      labelsOf[ResultRecord].productIterator.toList
        .map(_.asInstanceOf[String])
        .filterNot(_ === "canBeDisplayed")
    val inputHeaderFields = {
      val headline = os.read.lines(tracesInputFile).head
      val sep = Delimiter.fromPathUnsafe(componentExpectationFile)
      sep `split` headline
    }.toList
    val qcColumns = componentLabelColumns :+ QcPassColumn
    expColumnsUnfiltered shouldEqual {
      (if inputHeaderFields.head === "" then inputHeaderFields.tail
       else inputHeaderFields) ++ qcColumns
    }

    // Run all tests over both input files -- with or without index_col=0 -- as results should be invariant.
    forAll(Table("infile", tracesInputFile, tracesInputFileWithoutIndex)) {
      infile =>
        withTempDirectory { (tempdir: os.Path) =>
          val (expUnfilteredPath, expFilteredPath) =
            pretest(tempdir = tempdir, infile = infile)

          // With the pretest passed, run the action that generate outputs from inputs.
          runStandardWorkflow(
            defaultRoundsConfig,
            infile = infile,
            outfolder = tempdir
          )

          /* Now, first, just do basic existence check of files... */
          List(infile, expUnfilteredPath, expFilteredPath).map(
            os.isFile
          ) shouldEqual List(true, true, true)

          val obsLinesUnfiltered = os.read.lines(expUnfilteredPath)

          // ...then, check that no probes were excluded...
          obsLinesUnfiltered.length shouldEqual expLinesUnfiltered.length

          // ...now, use the observed and expected rows to do more sophisticated checks...
          withCsvPair(expUnfilteredPath, componentExpectationFile) {
            (obsRows: CsvRows, expRows: CsvRows) =>
              // The number of observed output rows should be the 1 less (header) than expected lines count.
              obsRows.size shouldEqual expLinesUnfiltered.length - 1

              // Each row should have the expected fields (column names from header).
              forAll(Table("fields", obsRows.toList*)) { row =>
                /* Observed row field names should equal expected row field names. */
                (
                  row.keySet -- expColumnsUnfiltered,
                  expColumnsUnfiltered.filterNot(row.keySet.contains)
                ) shouldEqual (Set(), List())
              }

              // Every QC result (component and aggregate) column should be as expected.
              forAll(Table("col", qcColumns*)) { col =>
                val obs = obsRows.map(_(col))
                val exp = expRows.map(_(col))
                obs shouldEqual exp
              }

              // Finally, do a stricter line-by-line equality test, for the UNFILTERED output.
              assertPairwiseEquality(
                observed = obsLinesUnfiltered.toList,
                expected = expLinesUnfiltered.toList
              )

              // Repeat the line-by-line equality test for the FILTERED output.
              val obsLinesFiltered = os.read.lines(expFilteredPath)
              val expLinesFiltered =
                os.read.lines(wholemealFilteredExpectationFile)
              assertPairwiseEquality(
                observed = obsLinesFiltered.toList,
                expected = expLinesFiltered.toList
              )
          }
        }
    }

  }

  test("Probe exclusion is correct.") {
    given [A] => Shrink[A] = Shrink.shrinkAny[A]

    // Taken from the timepoint column of the traces input file
    // Regional time 5 has rows where it's the "locus time" also, but regional time 4 doesn't have this.
    val expectedTimepoints = (0 to 9).toSet - 4

    forAll(
      Gen.someOf(expectedTimepoints),
      Gen.oneOf(tracesInputFile, tracesInputFileWithoutIndex)
    ) { (exclusions, infile) =>
      withCsvData(infile) { (rows: CsvRows) =>
        rows
          .map(_(ParserConfig.default.timeColumn))
          .toSet shouldEqual expectedTimepoints.map(_.toString)
      }
      val roundsConfig = {
        val seq = defaultRoundsConfig.sequence
        val loc = defaultRoundsConfig.locusGrouping
        val proxFilt = defaultRoundsConfig.proximityFilterStrategy
        ImagingRoundsConfiguration.unsafe(
          seq,
          loc,
          proxFilt,
          exclusions.map(ImagingTimepoint.unsafe).toSet,
          None,
          true
        )
      }
      withTempDirectory { (tempdir: os.Path) =>
        // Perform the pretest and get the expected result paths.
        val (expUnfilteredPath, expFilteredPath) =
          pretest(tempdir = tempdir, infile = infile)
        // Run the output-generating action under test.
        runStandardWorkflow(roundsConfig, infile = infile, outfolder = tempdir)
        // Now, first, just do basic existence check of files...
        List(infile, expUnfilteredPath, expFilteredPath).map(
          os.isFile
        ) shouldEqual List(true, true, true)

        if exclusions === expectedTimepoints then {
          /* Each file should be just a header. */
          withCsvData(expUnfilteredPath)((_: CsvRows).isEmpty shouldBe true)
          withCsvData(expFilteredPath)((_: CsvRows).isEmpty shouldBe true)
          os.read.lines(expUnfilteredPath).length shouldEqual 1
          os.read.lines(expFilteredPath).length shouldEqual 1
        } else {
          /* Each file should exhibit row-by-row equality with expectation. */
          withCsvPair(expUnfilteredPath, componentExpectationFile) {
            (obsRows: CsvRows, expRowsAll: CsvRows) =>
              assertPairwiseEquality(
                observed = obsRows.toList,
                expected = expRowsAll.toList
              )
          }
          // Just use strings here since Int-to-String is 1:1 (essentially), and this obviates need to unsafely lift String to Int.
          val discard = (r: Map[String, String]) =>
            exclusions.map(_.toString).toSet.contains(r("timepoint"))
          withCsvPair(expFilteredPath, wholemealFilteredExpectationFile) {
            (obsRows: CsvRows, expRowsAll: CsvRows) =>
              val expRowsFilt = expRowsAll.filterNot(discard)
              assertPairwiseEquality(
                observed = obsRows.toList,
                expected = expRowsFilt.toList
              )
          }
        }
      }
    }
  }

  test("Trace length filter is effective.") {
    implicit def noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

    val allLinesFilteredExpectation =
      os.read.lines(wholemealFilteredExpectationFile).toList
    val minLenDemarcation = 3
    val expWhenLessThanDemarcation = allLinesFilteredExpectation
    // Take the rows grouped together by (pos ID, region ID, roi ID) where group size is at least the demarcation size.
    // Drop the first 5 lines, which correspond to 1 for the header and 4 records before the block of 3 (pid=0, rid=23, tid=1).
    val expWhenEqualToDemarcation =
      allLinesFilteredExpectation.head :: allLinesFilteredExpectation
        .drop(5)
        .take(minLenDemarcation)
    val expWhenMoreThanDemarcation = List(allLinesFilteredExpectation.head)
    def genMinLenAndExp = Gen.oneOf(
      Gen
        .choose(0, 2)
        .map(NonnegativeInt.unsafe)
        .map(_ -> expWhenLessThanDemarcation),
      Gen.const(NonnegativeInt(3)).map(_ -> expWhenEqualToDemarcation),
      Gen
        .choose(4, Int.MaxValue)
        .map(NonnegativeInt.unsafe)
        .map(_ -> expWhenMoreThanDemarcation)
    )
    forAll(
      genMinLenAndExp,
      Gen.oneOf(tracesInputFile, tracesInputFileWithoutIndex)
    ) { case ((minTraceLen, expLinesFiltered), infile) =>
      withTempDirectory { (tempdir: os.Path) =>
        val (expUnfilteredPath, expFilteredPath) =
          pretest(tempdir = tempdir, infile = infile)
        // With the pretest passed, run the action that generate outputs from inputs.
        runStandardWorkflow(
          defaultRoundsConfig,
          infile = infile,
          outfolder = tempdir,
          minTraceLength = minTraceLen
        )
        // Now, first, just do basic existence check of files...
        List(infile, expUnfilteredPath, expFilteredPath).map(
          os.isFile
        ) shouldEqual List(true, true, true)

        /* Do a strict line-by-line equality test, for the UNFILTERED output; trace length filter has no effect. */
        assertPairwiseEquality(
          observed = os.read.lines(expUnfilteredPath).toList,
          expected = os.read.lines(componentExpectationFile).toList
        )

        /* Do a check of the lines of the filtered output file. */
        val observed = os.read.lines(expFilteredPath).toList
        assertPairwiseEquality(observed = observed, expected = expLinesFiltered)
      }
    }
  }

  /* Ancillary functions and types */
  type CsvRows = Iterable[Map[String, String]]
  private def componentExpectationFile = getResourcePath(
    "traces.labeled.unfiltered.csv"
  )
  private def wholemealFilteredExpectationFile = getResourcePath(
    "traces.labeled.filtered.csv"
  )
  private def getResourcePath(name: String): os.Path =
    os.Path(
      getClass.getResource(s"/TestLabelAndFilterLocusSpots/$name").getPath
    )

  private def pretest(tempdir: os.Path, infile: os.Path) = {
    val expUnfilteredPath = tempdir / s"${infile.baseName}.unfiltered.csv"
    val expFilteredPath = tempdir / s"${infile.baseName}.filtered.csv"
    os.isFile(tracesInputFile) shouldBe true // Input file exists.
    os.isFile(
      expUnfilteredPath
    ) shouldBe false // Unfiltered output doesn't yet exist.
    os.isFile(
      expFilteredPath
    ) shouldBe false // Filtered output doesn't yet exist.
    (expUnfilteredPath, expFilteredPath)
  }

  private def defaultRoundsConfig =
    ImagingRoundsConfiguration.unsafeFromJsonFile(
      getResourcePath("rounds_config.json")
    )
  private def tracesInputFile = getResourcePath("traces__with_index.raw.csv")
  private def tracesInputFileWithoutIndex = getResourcePath(
    "traces__without_index.raw.csv"
  )

  def assertPairwiseEquality[T: Equality](
      observed: List[T],
      expected: List[T]
  ) = {
    observed.length shouldEqual expected.length
    forAll(Table(("obs", "exp"), observed.zip(expected)*)) { case (obs, exp) =>
      obs shouldEqual exp
    }
  }

  private def runStandardWorkflow(
      roundsConfig: ImagingRoundsConfiguration,
      infile: os.Path = tracesInputFile,
      outfolder: os.Path,
      parserConfig: ParserConfig = ParserConfig.default,
      roiSize: RoiImageSize = RoiImageSize(
        PixelCountZ(PositiveInt(16)),
        PixelCountY(PositiveInt(32)),
        PixelCountX(PositiveInt(32))
      ),
      maxDistFromRegion: DistanceToRegion = DistanceToRegion(
        NonnegativeReal(800)
      ),
      minSignalToNoise: SignalToNoise = SignalToNoise(PositiveReal(2)),
      maxSigmaXY: SigmaXY = SigmaXY(PositiveReal(150)),
      maxSigmaZ: SigmaZ = SigmaZ(PositiveReal(400)),
      probeExclusions: List[ProbeName] = List(),
      minTraceLength: NonnegativeInt = NonnegativeInt(0)
  ) = {

    workflow(
      roiSize,
      roundsConfig,
      parserConfig,
      infile,
      maxDistFromRegion,
      minSignalToNoise,
      maxSigmaXY,
      maxSigmaZ,
      minTraceLength,
      analysisOutfolder = outfolder,
      pointsOutfolder = outfolder
    )
  }

  /** Use rows from a CSV file in arbitrary code. */
  private def withCsvData(
      filepath: os.Path
  )(code: Iterable[Map[String, String]] => Any): Any = {
    val reader = CSVReader.open(filepath.toIO)
    try { code(reader.allWithHeaders()) }
    finally { reader.close() }
  }

  /** Do arbitrary code with rows from a pair of CSV files. */
  private def withCsvPair(f1: os.Path, f2: os.Path)(
      code: (
          Iterable[Map[String, String]],
          Iterable[Map[String, String]]
      ) => Any
  ): Any = {
    var reader1: CSVReader = null
    val reader2 = CSVReader.open(f2.toIO)
    try {
      reader1 = CSVReader.open(f1.toIO)
      code(reader1.allWithHeaders(), reader2.allWithHeaders())
    } finally {
      if reader1 != null then { reader1.close() }
      reader2.close()
    }
  }
end TestLabelAndFilterLocusSpots
