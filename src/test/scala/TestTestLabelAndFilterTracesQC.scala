package at.ac.oeaw.imba.gerlich.looptrace

import scala.io.Source
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.funsuite.AnyFunSuite

import LabelAndFilterTracesQC.{ DistanceToRegion, SigmaXY, SigmaZ, SignalToNoise, ParserConfig, workflow }

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
            /* Relevant paths */
            val tracesFile = os.Path(getClass.getResource("/TestLabelAndFilterTracesQC/raw_trace_outfile.csv").getPath)
            val expFilteredPath = tempdir / s"${tracesFile.baseName}.filtered.csv"
            val expUnfilteredPath = tempdir / s"${tracesFile.baseName}.unfiltered.csv"
            
            /* Pretests */
            os.isFile(tracesFile) shouldBe true // Input file exists.
            os.isFile(expUnfilteredPath) shouldBe false // Unfiltered output doesn't yet exist.
            os.isFile(expFilteredPath) shouldBe false // Filtered output doesn't yet exist.
            
            workflow(
                ParserConfig.default, 
                tracesFile, 
                maxDistFromRegion = DistanceToRegion(NonnegativeReal(800)), 
                minSignalToNoise = SignalToNoise(PositiveReal(2)), 
                maxSigmaXY = SigmaXY(PositiveReal(150)), 
                maxSigmaZ = SigmaZ(PositiveReal(400)),
                probeExclusions = List(), 
                minTraceLength = NonnegativeInt(0), 
                outfolder = tempdir
                )

            /* Check existence of files. */
            os.isFile(tracesFile) shouldBe true // Original file is still present.
            os.isFile(expUnfilteredPath) shouldBe true // Unfiltered output now exists.
            os.isFile(expFilteredPath) shouldBe true // Filtered output now exists.
            
        }
    }

    test("Maximum-distance-to-region filter is effective.") { pending }

    test("SigmaXY filter is effective.") { pending }

    test("SigmaZ filter is effective.") { pending }

    test("X dimension filter is effective.") { pending }

    test("Y dimension filter is effective.") { pending }

    test("Z dimension filter is effective.") { pending }

    test("Signal-to-noise filter is effective.") { pending }

    test("Trace length filter is effective.") { pending }

    test("Probe exclusion is effective; ignored probes are present in unfiltered--but not filtered--output file.") { pending }

end TestLabelAndFilterTracesQC
