package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ NotGiven, Random, Try }
import cats.*
import cats.data.{ NonEmptyList, NonEmptySet }
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*

import at.ac.oeaw.imba.gerlich.looptrace.collections.*
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.LabelAndFilterRois.*
import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.safeReadAllWithOrderedHeaders


/** Tests for the filtration of the individual supports (single FISH probes) of chromatin fiber traces */
class TestLabelAndFilterRois extends AnyFunSuite, DistanceSuite, LooptraceSuite, ScalacheckSuite, should.Matchers:

    test("Drift file is required (so that distance/proximity comparisons can use drift-corrected coordinates.)") {
        withTempDirectory{ (tmpdir: os.Path) => 
            // missing driftFile
            assertTypeError{ "workflow( " +
                "spotsFile = tmpdir / \"traces.csv\", " + 
                "proximityFilterStrategy = ImagingRoundsConfiguration.UniversalProximityPermission, " + 
                "unfilteredOutputFile = UnfilteredOutputFile.fromPath(tmpdir / \"unfiltered.csv\"), " + 
                "filteredOutputFile = FilteredOutputFile.fromPath(tmpdir / \"filtered.csv\"), " + 
                "extantOutputHandler = ExtantOutputHandler.Overwrite " + 
                ")"
            }
            // Add in driftFile to get compilation.
            assertCompiles{ "workflow( " +
                "spotsFile = tmpdir / \"traces.csv\", " + 
                "driftFile = tmpdir / \"drift.csv\", " + 
                "proximityFilterStrategy = ImagingRoundsConfiguration.UniversalProximityPermission, " + 
                "unfilteredOutputFile = UnfilteredOutputFile.fromPath(tmpdir / \"unfiltered.csv\"), " + 
                "filteredOutputFile = FilteredOutputFile.fromPath(tmpdir / \"filtered.csv\"), " + 
                "extantOutputHandler = ExtantOutputHandler.Overwrite " + 
                ")"
            }
        }
    }

    test("With UNIVERSAL PROHIBITIVE grouping, spot distance comparison is accurate, uses drift correction--coarse, fine, or both--can switch distance measure (#146), and is invariant under order of drifts.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        
        // Generate permutations of lines to test invariance under input order.
        def genLinesPermutation(linesBlock: String): Gen[Array[String]] = {
            val lines = linesBlock.stripMargin.split("\n")
            Gen.const(lines.head +: Random.shuffle(lines.tail).toArray)
        }

        // The ROIs data for each test iteration
        val spotsText = """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
            |0,P0001.zarr,27,0,18,104,1052,10,26,88,120,1036,1068
            |1,P0001.zarr,27,0,18,1739,264,10,26,1723,1755,248,280
            |2,P0001.zarr,27,0,3,1878,314,0,11,1870,1886,47,347
            |3,P0001.zarr,28,0,5.5,1380,1457,-1,14,1364,1396,1441,1473
            |4,P0001.zarr,28,0,7,1378,1459.5,0,15,8,40,1343,1375
            |5,P0001.zarr,28,0,10,1783,1084,2,18,1767,1799,1068,1100
            |6,P0001.zarr,29,0,10,589,1799,2,18,433,465,1283,1315
            |7,P0001.zarr,29,0,12,588,1779,2,18,572,604,1763,1795
            |8,P0001.zarr,29,0,11,595,1780,1,17,679,711,564,596
            |9,P0001.zarr,29,0,11,993,1721,3,19,977,1009,1705,1737
            |10,P0001.zarr,30,0,10,1783,1084,2,18,1767,1799,1068,1100
            |11,P0001.zarr,30,0,10.1,549,1280.8,2,18,433,465,1283,1315
            |12,P0001.zarr,30,0,10,548.5,1280.6,2,18,572,604,1763,1795
            |13,P0001.zarr,30,0,9,995,1780,1,17,679,711,564,596
            |14,P0001.zarr,30,0,8,993.2,1781,3,19,977,1009,1705,1737
            |15,P0001.zarr,30,0,14.4,589.5,1779.3,2,18,572,604,1763,1795
            |"""
        
        // Drift file data
        object DriftFileTexts:
            val allZero = """,frame,position,z_px_coarse,y_px_coarse,x_px_coarse,z_px_fine,y_px_fine,x_px_fine
                |0,27,P0001.zarr,0,0,0,0,0,0
                |1,28,P0001.zarr,0,0,0,0,0,0
                |2,29,P0001.zarr,0.0,0,0,0,0,0
                |3,30,P0001.zarr,0.0,0,0,0,0,0
                |"""
            val nonZero = """,frame,position,z_px_coarse,y_px_coarse,x_px_coarse,z_px_fine,y_px_fine,x_px_fine
                |0,27,P0001.zarr,-2.0,8.0,-24.0,0.3048142040458287,0.2167426082715708,0.46295638298323727
                |1,28,P0001.zarr,2.0,4.0,-20.0,0.6521556133243969,-0.32279031643811845,0.8467576764912169
                |2,29,P0001.zarr,0.0,6.0,-16.0,-0.32831460930799267,0.5707716296861373,0.768359957646404
                |3,30,P0001.zarr,-2.0,2.0,-12.0,-0.6267951175716121,0.24476613641147094,0.5547602737043816
                |"""
            val zeroCoarse = """,frame,position,z_px_coarse,y_px_coarse,x_px_coarse,z_px_fine,y_px_fine,x_px_fine
                |0,27,P0001.zarr,0.0,0.0,0.0,0.3048142040458287,0.2167426082715708,0.46295638298323727
                |1,28,P0001.zarr,0.0,0.0,0.0,0.6521556133243969,-0.32279031643811845,0.8467576764912169
                |2,29,P0001.zarr,0.0,0.0,0.0,-0.32831460930799267,0.5707716296861373,0.768359957646404
                |3,30,P0001.zarr,0.0,0.0,0.0,-0.6267951175716121,0.24476613641147094,0.5547602737043816
                |"""
            val zeroFine = """,frame,position,z_px_coarse,y_px_coarse,x_px_coarse,z_px_fine,y_px_fine,x_px_fine
                |0,27,P0001.zarr,-4.0,8.0,-24.0,0,0,0
                |1,28,P0001.zarr,-2.0,4.0,-20.0,0,0,0
                |2,29,P0001.zarr,2.0,6.0,-16.0,0,0,0
                |3,30,P0001.zarr,4.0,2.0,-12.0,0,0,0
                |"""
        end DriftFileTexts

        // Infinitely large or small definition of proximity
        val extremeArguments: List[(String, PositiveReal, String)] = {
            val almostZero = PositiveReal(1e-323)
            val first = (
                DriftFileTexts.allZero, 
                almostZero, 
                """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
                |0,P0001.zarr,27,0,18,104,1052,10,26,88,120,1036,1068
                |1,P0001.zarr,27,0,18,1739,264,10,26,1723,1755,248,280
                |2,P0001.zarr,27,0,3,1878,314,0,11,1870,1886,47,347
                |3,P0001.zarr,28,0,5.5,1380,1457,-1,14,1364,1396,1441,1473
                |4,P0001.zarr,28,0,7,1378,1459.5,0,15,8,40,1343,1375
                |6,P0001.zarr,29,0,10,589,1799,2,18,433,465,1283,1315
                |7,P0001.zarr,29,0,12,588,1779,2,18,572,604,1763,1795
                |8,P0001.zarr,29,0,11,595,1780,1,17,679,711,564,596
                |9,P0001.zarr,29,0,11,993,1721,3,19,977,1009,1705,1737
                |11,P0001.zarr,30,0,10.1,549,1280.8,2,18,433,465,1283,1315
                |12,P0001.zarr,30,0,10,548.5,1280.6,2,18,572,604,1763,1795
                |13,P0001.zarr,30,0,9,995,1780,1,17,679,711,564,596
                |14,P0001.zarr,30,0,8,993.2,1781,3,19,977,1009,1705,1737
                |15,P0001.zarr,30,0,14.4,589.5,1779.3,2,18,572,604,1763,1795
                |""",
                )
            val second = (
                DriftFileTexts.allZero, 
                PositiveReal(Double.MaxValue),
                headSpotsFile,
                )
            val theRest = for {
                drift <- List(
                    DriftFileTexts.nonZero, 
                    DriftFileTexts.zeroCoarse, 
                    DriftFileTexts.zeroFine,
                    )
                (value, expected) <- List(
                    // When min separation is vanishingly small, only coincident points are proximal; all else is kept.
                    almostZero -> spotsText, 
                    // When threshold is infinitely large, everything is proximal and nothing is kept.
                    PositiveReal(Double.MaxValue) -> headSpotsFile,
                    )
            } yield (drift, value, expected)
            first :: second :: theRest
        }
        
        // Pairs of threshold and (filtered) output expectation under zero drift
        val zeroDriftArguments =
            List(
                PositiveReal(1.0) -> 
                """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
                |0,P0001.zarr,27,0,18,104,1052,10,26,88,120,1036,1068
                |1,P0001.zarr,27,0,18,1739,264,10,26,1723,1755,248,280
                |2,P0001.zarr,27,0,3,1878,314,0,11,1870,1886,47,347
                |3,P0001.zarr,28,0,5.5,1380,1457,-1,14,1364,1396,1441,1473
                |4,P0001.zarr,28,0,7,1378,1459.5,0,15,8,40,1343,1375
                |6,P0001.zarr,29,0,10,589,1799,2,18,433,465,1283,1315
                |7,P0001.zarr,29,0,12,588,1779,2,18,572,604,1763,1795
                |8,P0001.zarr,29,0,11,595,1780,1,17,679,711,564,596
                |9,P0001.zarr,29,0,11,993,1721,3,19,977,1009,1705,1737
                |13,P0001.zarr,30,0,9,995,1780,1,17,679,711,564,596
                |14,P0001.zarr,30,0,8,993.2,1781,3,19,977,1009,1705,1737
                |15,P0001.zarr,30,0,14.4,589.5,1779.3,2,18,572,604,1763,1795
                |""",
                PositiveReal(2.0) -> 
                """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
                |0,P0001.zarr,27,0,18,104,1052,10,26,88,120,1036,1068
                |1,P0001.zarr,27,0,18,1739,264,10,26,1723,1755,248,280
                |2,P0001.zarr,27,0,3,1878,314,0,11,1870,1886,47,347
                |3,P0001.zarr,28,0,5.5,1380,1457,-1,14,1364,1396,1441,1473
                |4,P0001.zarr,28,0,7,1378,1459.5,0,15,8,40,1343,1375
                |6,P0001.zarr,29,0,10,589,1799,2,18,433,465,1283,1315
                |7,P0001.zarr,29,0,12,588,1779,2,18,572,604,1763,1795
                |8,P0001.zarr,29,0,11,595,1780,1,17,679,711,564,596
                |9,P0001.zarr,29,0,11,993,1721,3,19,977,1009,1705,1737
                |15,P0001.zarr,30,0,14.4,589.5,1779.3,2,18,572,604,1763,1795
                |""", 
                PositiveReal(3.0) -> 
                """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
                |0,P0001.zarr,27,0,18,104,1052,10,26,88,120,1036,1068
                |1,P0001.zarr,27,0,18,1739,264,10,26,1723,1755,248,280
                |2,P0001.zarr,27,0,3,1878,314,0,11,1870,1886,47,347
                |6,P0001.zarr,29,0,10,589,1799,2,18,433,465,1283,1315
                |8,P0001.zarr,29,0,11,595,1780,1,17,679,711,564,596
                |9,P0001.zarr,29,0,11,993,1721,3,19,977,1009,1705,1737
                |""",
            ).map((t, exp) => (DriftFileTexts.allZero, t, exp))
        
        val fineDriftOnlyArguments = List(
            PositiveReal(0.25) -> spotsText,
            PositiveReal(0.75) ->  
            """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
            |0,P0001.zarr,27,0,18,104,1052,10,26,88,120,1036,1068
            |1,P0001.zarr,27,0,18,1739,264,10,26,1723,1755,248,280
            |2,P0001.zarr,27,0,3,1878,314,0,11,1870,1886,47,347
            |3,P0001.zarr,28,0,5.5,1380,1457,-1,14,1364,1396,1441,1473
            |4,P0001.zarr,28,0,7,1378,1459.5,0,15,8,40,1343,1375
            |5,P0001.zarr,28,0,10,1783,1084,2,18,1767,1799,1068,1100
            |6,P0001.zarr,29,0,10,589,1799,2,18,433,465,1283,1315
            |7,P0001.zarr,29,0,12,588,1779,2,18,572,604,1763,1795
            |8,P0001.zarr,29,0,11,595,1780,1,17,679,711,564,596
            |9,P0001.zarr,29,0,11,993,1721,3,19,977,1009,1705,1737
            |10,P0001.zarr,30,0,10,1783,1084,2,18,1767,1799,1068,1100
            |13,P0001.zarr,30,0,9,995,1780,1,17,679,711,564,596
            |14,P0001.zarr,30,0,8,993.2,1781,3,19,977,1009,1705,1737
            |15,P0001.zarr,30,0,14.4,589.5,1779.3,2,18,572,604,1763,1795
            |""",
            PositiveReal(6.0) -> 
            """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
            |0,P0001.zarr,27,0,18,104,1052,10,26,88,120,1036,1068
            |1,P0001.zarr,27,0,18,1739,264,10,26,1723,1755,248,280
            |2,P0001.zarr,27,0,3,1878,314,0,11,1870,1886,47,347
            |6,P0001.zarr,29,0,10,589,1799,2,18,433,465,1283,1315
            |9,P0001.zarr,29,0,11,993,1721,3,19,977,1009,1705,1737
            |""",
        ).map((t, exp) => (DriftFileTexts.zeroCoarse, t, exp))

        val coarseDriftOnlyArguments = List(
            PositiveReal(1.0) -> 
            """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
            |0,P0001.zarr,27,0,18,104,1052,10,26,88,120,1036,1068
            |1,P0001.zarr,27,0,18,1739,264,10,26,1723,1755,248,280
            |2,P0001.zarr,27,0,3,1878,314,0,11,1870,1886,47,347
            |3,P0001.zarr,28,0,5.5,1380,1457,-1,14,1364,1396,1441,1473
            |4,P0001.zarr,28,0,7,1378,1459.5,0,15,8,40,1343,1375
            |5,P0001.zarr,28,0,10,1783,1084,2,18,1767,1799,1068,1100
            |6,P0001.zarr,29,0,10,589,1799,2,18,433,465,1283,1315
            |7,P0001.zarr,29,0,12,588,1779,2,18,572,604,1763,1795
            |8,P0001.zarr,29,0,11,595,1780,1,17,679,711,564,596
            |9,P0001.zarr,29,0,11,993,1721,3,19,977,1009,1705,1737
            |10,P0001.zarr,30,0,10,1783,1084,2,18,1767,1799,1068,1100
            |13,P0001.zarr,30,0,9,995,1780,1,17,679,711,564,596
            |14,P0001.zarr,30,0,8,993.2,1781,3,19,977,1009,1705,1737
            |15,P0001.zarr,30,0,14.4,589.5,1779.3,2,18,572,604,1763,1795
            |""", 
            PositiveReal(5.0) -> 
            """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
            |0,P0001.zarr,27,0,18,104,1052,10,26,88,120,1036,1068
            |1,P0001.zarr,27,0,18,1739,264,10,26,1723,1755,248,280
            |2,P0001.zarr,27,0,3,1878,314,0,11,1870,1886,47,347
            |5,P0001.zarr,28,0,10,1783,1084,2,18,1767,1799,1068,1100
            |6,P0001.zarr,29,0,10,589,1799,2,18,433,465,1283,1315
            |8,P0001.zarr,29,0,11,595,1780,1,17,679,711,564,596
            |9,P0001.zarr,29,0,11,993,1721,3,19,977,1009,1705,1737
            |10,P0001.zarr,30,0,10,1783,1084,2,18,1767,1799,1068,1100
            |""", 
        ).map((t, exp) => (DriftFileTexts.zeroFine, t, exp))

        val bothDriftArguments = List(
            PositiveReal(3.0) -> 
            """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
            |0,P0001.zarr,27,0,18,104,1052,10,26,88,120,1036,1068
            |1,P0001.zarr,27,0,18,1739,264,10,26,1723,1755,248,280
            |2,P0001.zarr,27,0,3,1878,314,0,11,1870,1886,47,347
            |5,P0001.zarr,28,0,10,1783,1084,2,18,1767,1799,1068,1100
            |6,P0001.zarr,29,0,10,589,1799,2,18,433,465,1283,1315
            |7,P0001.zarr,29,0,12,588,1779,2,18,572,604,1763,1795
            |8,P0001.zarr,29,0,11,595,1780,1,17,679,711,564,596
            |9,P0001.zarr,29,0,11,993,1721,3,19,977,1009,1705,1737
            |10,P0001.zarr,30,0,10,1783,1084,2,18,1767,1799,1068,1100
            |15,P0001.zarr,30,0,14.4,589.5,1779.3,2,18,572,604,1763,1795
            |""", 
            PositiveReal(17.0) -> 
            """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
            |0,P0001.zarr,27,0,18,104,1052,10,26,88,120,1036,1068
            |1,P0001.zarr,27,0,18,1739,264,10,26,1723,1755,248,280
            |2,P0001.zarr,27,0,3,1878,314,0,11,1870,1886,47,347
            |9,P0001.zarr,29,0,11,993,1721,3,19,977,1009,1705,1737
            |""", 
            PositiveReal(16.0) -> 
            """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
            |0,P0001.zarr,27,0,18,104,1052,10,26,88,120,1036,1068
            |1,P0001.zarr,27,0,18,1739,264,10,26,1723,1755,248,280
            |2,P0001.zarr,27,0,3,1878,314,0,11,1870,1886,47,347
            |6,P0001.zarr,29,0,10,589,1799,2,18,433,465,1283,1315
            |9,P0001.zarr,29,0,11,993,1721,3,19,977,1009,1705,1737
            |""", 
        ).map((t, exp) => (DriftFileTexts.nonZero, t, exp))

        forAll (Table(
            ("driftLines", "threshold", "expectOutput"), 
            (extremeArguments ::: zeroDriftArguments ::: fineDriftOnlyArguments ::: coarseDriftOnlyArguments ::: bothDriftArguments)*
            )) { (driftLines, threshold, expectOutput) => 
                // Reduce minSuccessful here since it's costly time-wise, and there are very few drift lines.
                forAll (genLinesPermutation(driftLines), arbitrary[ExtantOutputHandler], minSuccessful(5)) { (driftLines, handleOutput) =>
                    withTempDirectory{ (tmpdir: os.Path) => 
                        // Expected output is entirely determined by inputs.
                        val expLines = expectOutput.stripMargin.split("\n").toList
                        
                        /* Set up the input files and target outputs. */
                        val spotsFile = tmpdir / "rois.csv"
                        os.write(spotsFile, spotsText.stripMargin)
                        val driftFile = tmpdir / "drift.csv"
                        os.write(driftFile, driftLines.map(_ ++ "\n"))
                        val filteredFile = FilteredOutputFile.fromPath(tmpdir / "filteredOutput.csv")
                        val unfilteredFile = UnfilteredOutputFile.fromPath(tmpdir / "unfilteredOutput.csv")
                        
                        /* Pretest */
                        os.exists(filteredFile) shouldBe false
                        os.exists(unfilteredFile) shouldBe false
                        
                        // Make the call under test.
                        workflow(
                            spotsFile = spotsFile, 
                            driftFile = driftFile, 
                            proximityFilterStrategy = ImagingRoundsConfiguration.UniversalProximityProhibition(threshold),
                            filteredOutputFile = filteredFile, 
                            unfilteredOutputFile = unfilteredFile, 
                            extantOutputHandler = handleOutput,
                            )

                        /* Make the assertions. */
                        os.isFile(filteredFile) shouldBe true
                        os.isFile(unfilteredFile) shouldBe true
                        val obsLines = os.read.lines(filteredFile).toList
                        obsLines.length shouldEqual expLines.length
                        obsLines.zip(expLines).filter(_ =!= _) shouldEqual List()
                    }
                }
            }
    }
    
    test("Regardless of grouping semantic, any (position, time) repeat in drift file is an error, and drift must be fine not just coarse.") {
        val spotsLines = """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
            |0,P0001.zarr,27,0,18.594063700840934,104.97590586866923,1052.9315138200425,10.594063700840934,26.594063700840934,88.97590586866923,120.97590586866923,1036.9315138200425,1068.9315138200425
            |1,P0001.zarr,27,0,18.45511019130035,1739.9764501391553,264.9779910476261,10.45511019130035,26.45511019130035,1723.9764501391553,1755.9764501391553,248.97799104762612,280.9779910476261
            |""".stripMargin
        
        val coarseDriftLines = """,frame,position,z_px_coarse,y_px_coarse,x_px_coarse
            |0,0,P0001.zarr,-2.0,8.0,-24.0
            |1,1,P0001.zarr,-2.0,8.0,-20.0
            |2,0,P0002.zarr,0.0,2.0,-6.0
            |3,1,P0002.zarr,0.0,0.0,-4.0
            |""".stripMargin
        
        val fineDriftLines = """,frame,position,z_px_coarse,y_px_coarse,x_px_coarse,z_px_fine,y_px_fine,x_px_fine
            |0,0,P0001.zarr,-2.0,8.0,-24.0,0.3048142040458287,0.2167426082715708,0.46295638298323727
            |1,1,P0001.zarr,-2.0,8.0,-20.0,0.6521556133243969,-0.32279031643811845,0.8467576764912169
            |2,2,P0001.zarr,0.0,6.0,-16.0,-0.32831460930799267,0.5707716296861373,0.768359957646404
            |3,3,P0001.zarr,0.0,6.0,-12.0,-0.6267951175716121,0.24476613641147094,0.5547602737043816
            |4,2,P0001.zarr,0.0,4.0,-8.0,-0.6070769698700896,0.19949205544251802,1.006397174550077
            |5,4,P0001.zarr,0.0,2.0,-6.0,-0.47287036567793056,-0.14219950183651772,-0.038394014732762875
            |6,0,P0002.zarr,0.0,0.0,-6.0,-0.19920717991788411,0.6764858399077859,1.0163307592432946
            |7,0,P0002.zarr,0.0,0.0,-4.0,-0.042701396565681886,0.6483520746291591,-0.3969314534982328
            |8,0,P0002.zarr,0.0,0.0,-4.0,0.15150446478395588,0.3227722878990854,-0.24910778221670044
            |9,1,P0002.zarr,0.0,0.0,-4.0,0.25331586851532506,0.014229358683931789,0.0077311034198835745
            |""".stripMargin
        
        // NB: the frame value for the groping comes from the spotsLines variable.
        def genFilterStrategy = {
            val singleGroup = NonEmptyList.one(NonEmptySet.one(Timepoint(NonnegativeInt(27))))
            arbitrary[PositiveReal].flatMap{ threshold => Gen.oneOf(
                ImagingRoundsConfiguration.UniversalProximityPermission, 
                ImagingRoundsConfiguration.UniversalProximityProhibition(threshold), 
                ImagingRoundsConfiguration.SelectiveProximityPermission(threshold, singleGroup), 
                ImagingRoundsConfiguration.SelectiveProximityProhibition(threshold, singleGroup), 
            )}
        }

        forAll (Table(
            ("driftLines", "expected", "mapMessage"), 
            (coarseDriftLines, 1, "[0-9]+ error\\(s\\) converting drift file .+ rows to records!".r.findAllMatchIn(_: String).toList.length), 
            (fineDriftLines, s"2 repeated (pos, time) pairs: ${List(("P0001.zarr", 2) -> List(2, 4), ("P0002.zarr", 0) -> List(6, 7, 8))}", identity[String])
            )) { (driftLines, expected, mapMessage) => 
            forAll (genPosReal, arbitrary[ExtantOutputHandler], genFilterStrategy) {
                (threshold, outputHandler, grouping) =>
                    withTempDirectory{ (tmpdir: os.Path) => 
                        val spotsFile = tmpdir / "spots.csv"
                        os.write(spotsFile, spotsLines)
                        val driftFile = tmpdir / "drift.csv"
                        os.write(driftFile, driftLines)
                        val caught = intercept[Exception]{ workflow(
                            spotsFile = spotsFile, 
                            driftFile = driftFile, 
                            proximityFilterStrategy = grouping, 
                            filteredOutputFile = FilteredOutputFile.fromPath(tmpdir / "filtered.csv"),
                            unfilteredOutputFile = UnfilteredOutputFile.fromPath(tmpdir / "unfiltered.csv"),
                            extantOutputHandler = outputHandler,
                            )
                        }
                        val errMsg = caught.getMessage
                        mapMessage(errMsg) shouldEqual expected
                    }
            }
        }
    }

    test("Regardless of grouping semantic, the filtered output file is the unfiltered file minus the records with neighbors, and the neighbors column.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        /* Generate reasonable ROIs (controlling centroid and bounding box). */
        given arbMargin: Arbitrary[BoundingBox.Margin] = getArbForMargin(NonnegativeReal(1.0), NonnegativeReal(32.0))
        given arbPoint: Arbitrary[Point3D] = getArbForPoint3D(-2048.0, 2048.0)
        
        def genSpotsAndDriftsAndStrategy = {
            /* Generate drifts. */
            def genCoarseDrift: Gen[CoarseDrift] = {
                val genInt = Gen.choose(-100, 100)
                Gen.zip(genInt, genInt, genInt).map((z, y, x) => CoarseDrift(ZDir(z), YDir(y), XDir(x)))
            }
            def genFineDrift: Gen[FineDrift] = {
                val genX = Gen.choose[Double](-1, 1)
                Gen.zip(genX, genX, genX).map((z, y, x) => FineDrift(ZDir(z), YDir(y), XDir(x)))
            }
            for {
                // Generate at least 2 spots rows so that there's at least the chance to have the region timepoints in different groups.
                (spots, drifts) <- genSpotsAndDrifts(genCoarseDrift, genFineDrift).suchThat(_._1.length > 1)
                proxFilterStrategy <- {
                    val genThreshold = Gen.choose(1.0, 10.0).map(PositiveReal.unsafe)
                    Gen.oneOf(
                        Gen.const(ImagingRoundsConfiguration.UniversalProximityPermission), 
                        genThreshold.map(ImagingRoundsConfiguration.UniversalProximityProhibition.apply), 
                        genSpotCoveringGrouping(genThreshold)(spots.toList),
                        )
                }
            } yield (spots, drifts, proxFilterStrategy)
        }

        forAll (genSpotsAndDriftsAndStrategy, arbitrary[ExtantOutputHandler]) {
            case ((spots, driftRows, proxFilterStrategy), handleOutput) => 
                withTempDirectory{ (tmpdir: os.Path) => 
                    val spotsFile = tmpdir / "spots.csv"
                    os.write(spotsFile, getSpotsFileLines(spots.toList).map(_ ++ "\n"))
                    val driftFile = tmpdir / "drift.csv"
                    os.write(driftFile, getDriftFileLines(driftRows).map(_ ++ "\n"))
                    val unfilteredFile: UnfilteredOutputFile = UnfilteredOutputFile.fromPath(tmpdir / "unfiltered.csv")
                    val filteredFile: FilteredOutputFile = FilteredOutputFile.fromPath(tmpdir / "filtered.csv")
                    workflow(
                        spotsFile = spotsFile, 
                        driftFile = driftFile, 
                        proximityFilterStrategy = proxFilterStrategy, 
                        unfilteredOutputFile = unfilteredFile,
                        filteredOutputFile = filteredFile,
                        extantOutputHandler = handleOutput,
                        )
                    val (headUnfiltered, rowsUnfiltered) = safeReadAllWithOrderedHeaders(unfilteredFile) match {
                        case Left(e) => fail("Could not read unfiltered file: $e")
                        case Right((head, rows)) => head -> rows
                    }
                    val (headFiltered, rowsFiltered) = safeReadAllWithOrderedHeaders(filteredFile) match {
                        case Left(e) => fail("Could not read filtered file: $e")
                        case Right((head, rows)) => head -> rows
                    }
                    headUnfiltered shouldEqual headFiltered :+ "neighbors"
                    rowsFiltered shouldEqual rowsUnfiltered.filter(_("neighbors") === "").map(_ - "neighbors")
                }
        }
    }

    test("Regardless of proximity filtration strategy, any ROI without drift is an error. #196") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        /* Generate reasonable ROIs (controlling centroid and bounding box). */
        given arbMargin: Arbitrary[BoundingBox.Margin] = getArbForMargin(NonnegativeReal(1.0), NonnegativeReal(32.0))
        given arbPoint: Arbitrary[Point3D] = getArbForPoint3D(-2048.0, 2048.0)
        
        def genSpotsDriftsDropsAndFilterStrategy = {
            /* Generate drifts. */
            def genCoarseDrift: Gen[CoarseDrift] = {
                val genInt = Gen.choose(-10000, 10000)
                Gen.zip(genInt, genInt, genInt).map((z, y, x) => CoarseDrift(ZDir(z), YDir(y), XDir(x)))
            }
            def genFineDrift: Gen[FineDrift] = {
                val genX = Gen.choose[Double](-100, 100)
                Gen.zip(genX, genX, genX).map((z, y, x) => FineDrift(ZDir(z), YDir(y), XDir(x)))
            }
            for {
                // Generate at least 2 spots rows so that there's at least the chance to have the region timepoints in different groups.
                (spots, driftRows, numDropped) <- genSpotsAndDriftsWithDrop(genCoarseDrift, genFineDrift).suchThat(_._1.length > 1)
                proxFilterStrategy <- genSpotCoveringGrouping(arbitrary[PositiveReal])(spots.toList)
            } yield (spots, driftRows, numDropped, proxFilterStrategy)
        }

        forAll (genSpotsDriftsDropsAndFilterStrategy, arbitrary[PositiveReal], arbitrary[ExtantOutputHandler]) { 
            case ((spots, driftRows, numDropped, proxFilterStrategy), threshold, handleOutput) => 
                withTempDirectory{ (tmpdir: os.Path) => 
                    val spotsFile = tmpdir / "spots.csv"
                    os.write(spotsFile, getSpotsFileLines(spots.toList).map(_ ++ "\n"))
                    val driftFile = tmpdir / "drift.csv"
                    os.write(driftFile, getDriftFileLines(driftRows).map(_ ++ "\n"))
                    def call() = workflow(
                        spotsFile = spotsFile, 
                        driftFile = driftFile, 
                        proximityFilterStrategy = proxFilterStrategy, 
                        unfilteredOutputFile = UnfilteredOutputFile.fromPath(tmpdir / "unfiltered.csv"),
                        filteredOutputFile = FilteredOutputFile.fromPath(tmpdir / "filtered.csv"),
                        extantOutputHandler = handleOutput,
                        )
                    if numDropped === 0
                    then Try(call()).fold(err => fail(s"Expected success but failed: $err"), _ => succeed)
                    else assertThrows[DriftRecordNotFoundError](call())
                }
        }
    }

    test("Regardless of grouping semantic, processing doesn't alter ROIs: the collection of ROIs parsed from input is identical to the collection of ROIs parsed from unfiltered, labeled output.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        val time1 = 3
        val time2 = 4
        val time3 = 5
        val time4 = 6
        val timepoints = List(time1, time2, time3, time4).map(Timepoint.unsafe)
        val rawPosName = "P0001.zarr"
        val driftFileText = s""",frame,position,z_px_coarse,y_px_coarse,x_px_coarse,z_px_fine,y_px_fine,x_px_fine
            |0,${time1},${rawPosName},-2.0,8.0,-24.0,0.3048142040458287,0.2167426082715708,0.46295638298323727
            |1,${time2},${rawPosName},2.0,4.0,-20.0,0.6521556133243969,-0.32279031643811845,0.8467576764912169
            |2,${time3},${rawPosName},0.0,6.0,-16.0,-0.32831460930799267,0.5707716296861373,0.768359957646404
            |3,${time4},${rawPosName},-2.0,2.0,-12.0,-0.6267951175716121,0.24476613641147094,0.5547602737043816
            |""".stripMargin
        
        def genRegionalRound(t: Timepoint)(using arbProbe: Arbitrary[ProbeName]): Gen[RegionalImagingRound] = 
            arbitrary[ProbeName].map(RegionalImagingRound(t, _))

        def genLocalRound(t: Timepoint)(using arbProbe: Arbitrary[ProbeName]): Gen[LocusImagingRound] = 
            arbitrary[ProbeName].map(LocusImagingRound(t, _))

        given writerForRound: Writer[ImagingRound] = ImagingRound.rwForImagingRound

        // Choose from probe groupings available given the timepoints used in drift file.
        given arbGrouping: Arbitrary[ImagingRoundsConfiguration.ProximityFilterStrategy] = {
            val genGroups = Gen.oneOf(List(
                NonEmptyList.of(NonEmptySet.one(time1), NonEmptySet.one(time3), NonEmptySet.of(time2, time4)), 
                NonEmptyList.of(NonEmptySet.of(time1, time4), NonEmptySet.of(time2, time3)),
                NonEmptyList.of(NonEmptySet.of(time1, time2, time4), NonEmptySet.one(time3)), 
                NonEmptyList.one(NonEmptySet.of(time1, time2, time3, time4)),
                )
                .map(_.map(_.map(Timepoint.unsafe)))
            )
            
            val genThreshold = Gen.choose(1.0, 10000.0).map(PositiveReal.unsafe)
            Gen.oneOf(
                Gen.zip(genThreshold, genGroups).flatMap{ (t, g) => 
                    Gen.oneOf(
                        ImagingRoundsConfiguration.SelectiveProximityPermission(t, g), 
                        ImagingRoundsConfiguration.SelectiveProximityProhibition(t, g),
                        )
                }, 
                genThreshold.map(ImagingRoundsConfiguration.UniversalProximityProhibition.apply), 
                Gen.const(ImagingRoundsConfiguration.UniversalProximityPermission), 
            ).toArbitrary
        }

        /* Control the generation of ROIs to match the drift file text. */
        given arbPos: Arbitrary[PositionName] = Gen.const(PositionName(rawPosName)).toArbitrary
        given arbTimepoint: Arbitrary[Timepoint] = Gen.oneOf(timepoints).toArbitrary
        
        /* Generate reasonable ROIs (controlling centroid and bounding box). */
        given arbMargin: Arbitrary[BoundingBox.Margin] = getArbForMargin(NonnegativeReal(1.0), NonnegativeReal(32.0))
        given arbPoint: Arbitrary[Point3D] = getArbForPoint3D(-2048.0, 2048.0)

        forAll { (rois: List[RegionalBarcodeSpotRoi], proxFilerStrategy: ImagingRoundsConfiguration.ProximityFilterStrategy, handleOutput: ExtantOutputHandler) =>
            val inputLines = getSpotsFileLines(rois)
            withTempDirectory{ (tmpdir: os.Path) => 
                val spotsFile = tmpdir / "spots.csv"
                os.write(spotsFile, inputLines.map(_ ++ "\n"))
                val driftFile = tmpdir / "drift.csv"
                os.write(driftFile, driftFileText)
                val filtFile: FilteredOutputFile = FilteredOutputFile.fromPath(tmpdir / "filtered.csv")
                val unfiltFile: UnfilteredOutputFile = UnfilteredOutputFile.fromPath(tmpdir / "unfiltered.csv")
                workflow(
                    spotsFile = spotsFile, 
                    driftFile = driftFile, 
                    proximityFilterStrategy = proxFilerStrategy, 
                    unfilteredOutputFile = unfiltFile, 
                    filteredOutputFile = filtFile, 
                    extantOutputHandler = handleOutput,
                    )
                val (unfiltRecords, expectRecords) = {
                    val delim = Delimiter.CommaSeparator
                    os.read.lines(unfiltFile).map(delim.split(_).toList).toList -> inputLines.map(delim.split(_).toList)
                }
                unfiltRecords.map(_.dropRight(1)) shouldEqual expectRecords
            }
        }
    }

    test("PERMISSIVE semantic behaves as expected. #222") {
        given arbPoint: Arbitrary[Point3D] = getArbForPoint3D(-1, 1) // Limit to [-1, 1] in all dimensions.
        given ordReg: Ordering[Timepoint] = Order[Timepoint].toOrdering

        val regions = (0 to 3).map(RegionId.unsafe)
        val channel = Channel(NonnegativeInt(0))
        val margin = BoundingBox.Margin(NonnegativeReal(0.5))

        def genRois(lo: Int, hi: Int) = {
            def genOne = {
                def genPos = Gen.oneOf("P0001.zarr", "P0001.zarr").map(PositionName.apply)
                (arbitrary[RoiIndex], genPos, Gen.oneOf(regions), arbitrary[Point3D]).mapN((idx, pos, reg, pt) => 
                    val box = buildRectangularBox(pt)(margin, margin, margin)
                    RegionalBarcodeSpotRoi(idx, pos, reg, channel, pt, box)
                )
            }
            Gen.choose(lo, hi).flatMap(Gen.listOfN(_, genOne))
        }

        def genRoisAndStrategy(lo: Int, hi: Int) = for {
            rois <- genRois(lo, hi)
            obsReg = rois.map(_.region.get).toSet
            groups <- 
                if obsReg.size === 1 
                then Gen.const(NonEmptyList.one(obsReg.toNonEmptySetUnsafe))
                else Gen.choose(1, obsReg.size - 1).map{ k => 
                    val (g1, g2) = obsReg.toList.splitAt(k)
                    NonEmptyList.of(g1, g2).map(_.toSet.toNonEmptySetUnsafe)
                }
            threshold <- Gen.choose(1e-323, 1.0).map(PositiveReal.unsafe)
        } yield (rois, ImagingRoundsConfiguration.SelectiveProximityPermission(threshold, groups))

        forAll (genRoisAndStrategy(10, 50), minSuccessful(1000)) { (rois, proxFilterStrategy) =>
            val roisWithLine = NonnegativeInt.indexed(rois)
            val roiByLine = roisWithLine.map(_.swap).toMap
            buildNeighboringRoisFinder(roisWithLine, proxFilterStrategy) match {
                case Left(msg) => fail(s"Expected successful neighbors assignment, but got error: $msg")
                case Right(obsNeighbors) => 
                    val timeByLine = roisWithLine.map((r, i) => i -> r.region.get).toMap
                    def getGroup(line: LineNumber): Int = {
                        val time = timeByLine(line)
                        proxFilterStrategy.grouping.zipWithIndex.filter((g, _) => g.contains(time)) match {
                            case (_, i) :: Nil => i
                            case Nil => throw new Exception(s"No group found for time $time! Grouping: ${proxFilterStrategy.grouping}")
                            case _ => throw new Exception(s"Multiple groups found for time $time! Grouping: ${proxFilterStrategy.grouping}")
                        }
                    }
                    val unexpectedNeighbors = obsNeighbors.foldRight(List.empty[(RegionalBarcodeSpotRoi, NonEmptyList[RegionalBarcodeSpotRoi])]){ 
                        case ((line, neighborLines), acc) =>
                            val keyGroupIdx = getGroup(line)
                            neighborLines.toList
                                .filter(getGroup(_) === keyGroupIdx)
                                .toNel match{
                                    case None => acc
                                    case Some(ls) => (roiByLine(line), ls.map(roiByLine)) :: acc
                                }
                        }
                    unexpectedNeighbors shouldEqual List()
            }
        }
    }

    // This tests for both the ability to specify nothing for the grouping, and for the correctness of the definition of the partitioning (trivial) when no grouping is specified.
    test("For PROHIBITIVE (universal or selective) spot grouping, neighbor discovery is correct. #147") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        // Generate a reasonable margin on side of each centroid coordinate for ROI bounding boxes.
        given arbMargin: Arbitrary[BoundingBox.Margin] = getArbForMargin(NonnegativeReal(1.0), NonnegativeReal(32.0))

        /* Collection of points to consider for proximity-based filtration */
        val pt1 = Point3D(XCoordinate(-1.0), YCoordinate(1.0), ZCoordinate(2.0))
        val pt2 = Point3D(XCoordinate(-0.5), YCoordinate(0.5), ZCoordinate(3.0))
        val pt3 = Point3D(XCoordinate(-2.2), YCoordinate(-1.2), ZCoordinate(1.0))
        val pt4 = Point3D(XCoordinate(-1.2), YCoordinate(0.7), ZCoordinate(0.0))
        val pt5 = Point3D(XCoordinate(0.0), YCoordinate(-3.2), ZCoordinate(-2.0))

        /* Inputs and expected outputs */
        val inputTable = Table(
            ("threshold", "pointTimePairs", "grouping", "expected"), 
            (
                PositiveReal(1.5),
                List(pt1 -> 2, pt2 -> 3, pt3 -> 3, pt4 -> 1, pt5 -> 1), 
                List(NonEmptySet.of(2, 3), NonEmptySet.one(1)), // all proximal points in same group
                Map(0 -> NonEmptySet.one(1), 1 -> NonEmptySet.one(0)),
            ),
            (
                PositiveReal(1.5),
                List(pt1 -> 2, pt2 -> 3, pt3 -> 3, pt4 -> 1, pt5 -> 1), 
                List(NonEmptySet.one(2), NonEmptySet.of(1, 3)), // proximal points no longer grouped together
                Map.empty[Int, NonEmptySet[Int]],
            ),
            (
                PositiveReal(1.5),
                List(pt1 -> 2, pt2 -> 3, pt3 -> 3, pt4 -> 1, pt5 -> 1), // proximal points no longer grouped together
                List(NonEmptySet.one(3), NonEmptySet.of(1, 2)), 
                Map.empty[Int, NonEmptySet[Int]],
            ),
            (
                PositiveReal(Double.PositiveInfinity), 
                List(pt1 -> 1, pt2 -> 2, pt3 -> 3, pt4 -> 4, pt5 -> 5), 
                List(NonEmptySet.of(1, 2, 3, 4), NonEmptySet.one(5)),
                Map(
                    0 -> NonEmptySet.of(1, 2, 3), 
                    1 -> NonEmptySet.of(0, 2, 3), 
                    2 -> NonEmptySet.of(0, 1, 3), 
                    3 -> NonEmptySet.of(0, 1, 2), 
                ),
            ),
            (
                PositiveReal(Double.PositiveInfinity), 
                List(pt1 -> 1, pt2 -> 2, pt3 -> 3, pt4 -> 4, pt5 -> 5), 
                List(NonEmptySet.of(3, 4), NonEmptySet.of(1, 2, 5)),
                Map(
                    0 -> NonEmptySet.of(1, 4), 
                    1 -> NonEmptySet.of(0, 4), 
                    2 -> NonEmptySet.one(3), 
                    3 -> NonEmptySet.one(2), 
                    4 -> NonEmptySet.of(0, 1),
                ),
            ),
            (
                PositiveReal(2.5), 
                List(pt1 -> 11, pt2 -> 22, pt3 -> 33, pt4 -> 44, pt5 -> 55), 
                List(),
                Map(
                    0 -> NonEmptySet.of(1, 2, 3), 
                    1 -> NonEmptySet.of(0, 2), 
                    2 -> NonEmptySet.of(0, 1, 3), 
                    3 -> NonEmptySet.of(0, 2), 
                ),
            ),
            (
                PositiveReal(2.5), 
                List(pt1 -> 11, pt2 -> 22, pt3 -> 33, pt4 -> 44, pt5 -> 55), 
                List(NonEmptySet.of(22, 44), NonEmptySet.one(11), NonEmptySet.of(33, 55)),
                Map(),
            ),
            (
                PositiveReal(2.5), 
                List(pt1 -> 11, pt2 -> 22, pt3 -> 33, pt4 -> 44, pt5 -> 55), 
                List(NonEmptySet.of(22, 55), NonEmptySet.one(11), NonEmptySet.of(33, 44)),
                Map(2 -> NonEmptySet.one(3), 3 -> NonEmptySet.one(2)),
            ),
            )
        forAll (inputTable) { (threshold, pointTimePairs, rawGrouping, rawExpectation) => 
            def genRois: Gen[List[RegionalBarcodeSpotRoi]] = for {
                posName <- arbitrary[PositionName]
                ch <- arbitrary[Channel]
                rois <- NonnegativeInt.indexed(pointTimePairs).traverse{ case ((pt, time), i) => 
                    arbitrary[(BoundingBox.Margin, BoundingBox.Margin, BoundingBox.Margin)].map{ (offX, offY, offZ) => 
                        val intvX = buildInterval(pt.x, offX)(XCoordinate.apply)
                        val intvY = buildInterval(pt.y, offY)(YCoordinate.apply)
                        val intvZ = buildInterval(pt.z, offZ)(ZCoordinate.apply)
                        val bbox = BoundingBox(intvX, intvY, intvZ)
                        RegionalBarcodeSpotRoi(RoiIndex(i), posName, RegionId.unsafe(time), ch, pt, bbox)
                    }
                }
            } yield rois
            forAll (genRois) { rois => 
                val grouping = rawGrouping.toNel match {
                    case None => ImagingRoundsConfiguration.UniversalProximityProhibition(threshold)
                    case Some(grouping) => 
                        ImagingRoundsConfiguration.SelectiveProximityProhibition(threshold, grouping.map(_.map(Timepoint.unsafe)))
                }
                buildNeighboringRoisFinder(NonnegativeInt.indexed(rois), grouping) match {
                    case Left(err) => fail(s"Expected success but got error: $err")
                    case Right(observation) => 
                        val expectation = rawExpectation.map{ (k, vs) => NonnegativeInt.unsafe(k) -> vs.map(NonnegativeInt.unsafe) }
                        observation shouldEqual expectation
                }
            }
        }
    }

    test("Regardless of grouping semantic, it's a partition: A probe grouping that is NONEMPTY but does NOT COVER regional barcodes set is an error.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        // Generate a reasonable margin on side of each centroid coordinate for ROI bounding boxes.
        given arbPoint: Arbitrary[Point3D] = getArbForPoint3D(-2048.0, 2048.0)
        given arbMargin: Arbitrary[BoundingBox.Margin] = getArbForMargin(NonnegativeReal(1.0), NonnegativeReal(32.0))
        
        def genSmallRoisAndGrouping: Gen[(List[RegionalBarcodeSpotRoi], NonEmptyList[Timepoint], ImagingRoundsConfiguration.NontrivialProximityFilter)] = for {
            // First, generate ROIs timepoints, such that they're few in number (so quicker test), and
            // the number of unique timepoints is at least 2 (at least 1 to be uncovered by the grouping).
            rois <- {
                given arbTime: Arbitrary[Timepoint] = Gen.oneOf(List(7, 8, 9).map(Timepoint.unsafe)).toArbitrary
                Gen.choose(2, 10).flatMap(Gen.listOfN(_, arbitrary[RegionalBarcodeSpotRoi]))
            }.suchThat(_.map(_.time).toSet.size > 1)
            times = rois.map(_.time).toSet
            numGroups <- Gen.choose(1, times.size)
            rawFullGrouping <- Gen.oneOf(collections.partition(numGroups, times))
            (skipped, grouping) <- rawFullGrouping
                .traverse(_.toList.traverse(x => arbitrary[Boolean].map(_.either(x, x))))
                .map(_.foldLeft(List.empty[Timepoint] -> List.empty[NonEmptySet[Timepoint]]){ 
                    // Collect the timepoints to skip, and build up the probe/timepoint grouping.
                    case ((drops, acc), subMaybes) => 
                        val (newSkips, newKeeps) = Alternative[List].separate(subMaybes)
                        (newSkips ::: drops, newKeeps.toNel.fold(acc){ sub => sub.toNes :: acc })
                })
                .suchThat{ (skips, group) => skips.nonEmpty && group.nonEmpty } // At least 1 time is skipped, and grouping is nontrivial.
                .map(_.bimap(_.toNel.get, _.toNel.get))                         // safe b/c of .suchThat(...) filter
            threshold <- arbitrary[PositiveReal]
            proxFilterStrategy <- Gen.oneOf(
                ImagingRoundsConfiguration.SelectiveProximityPermission(threshold, grouping),
                ImagingRoundsConfiguration.SelectiveProximityProhibition(threshold, grouping),
                )
        } yield (rois, skipped, proxFilterStrategy)
        
        forAll (genSmallRoisAndGrouping) { (rois, uncoveredTimepoints, proxFilterStrategy) =>
            buildNeighboringRoisFinder(NonnegativeInt.indexed(rois), proxFilterStrategy) match {
                case Left(obsErrMsg) => 
                    val numGroupless = rois.filter{ r => uncoveredTimepoints.toNes.contains(r.time) }.length
                    val timesText = uncoveredTimepoints.map(_.get).toList.sorted.mkString(", ")
                    val expErrMsg = s"$numGroupless ROIs without timepoint declared in grouping. ${uncoveredTimepoints.size} undeclared timepoints: $timesText"
                    obsErrMsg shouldEqual expErrMsg
                case Right(_) => fail("Expected error message about invalid partition (non-covering), but didn't get it.")
            }
        }
    }

    test("Regardless of grouping semantic, it's a partition: A probe grouping that is NOT DISJOINT (timepoint repeated between groups) is an ERROR.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]

        // Generate a reasonable margin on side of each centroid coordinate for ROI bounding boxes.
        given arbPoint: Arbitrary[Point3D] = getArbForPoint3D(-2048.0, 2048.0)
        given arbMargin: Arbitrary[BoundingBox.Margin] = getArbForMargin(NonnegativeReal(1.0), NonnegativeReal(32.0))
        
        extension [X : Order](xs: Set[X])
            def unsafeToNes: NonEmptySet[X] = xs.toList.toNel.get.toNes

        def genSmallRoisAndGrouping: Gen[(List[RegionalBarcodeSpotRoi], NonEmptySet[Timepoint], ImagingRoundsConfiguration.NontrivialProximityFilter)] = for {
            // First, generate ROIs timepoints, such that they're few in number (so quicker test).
            rois <- {
                given arbTime: Arbitrary[Timepoint] = Gen.oneOf(List(7, 8, 9).map(Timepoint.unsafe)).toArbitrary
                Gen.choose(1, 10).flatMap(Gen.listOfN(_, arbitrary[RegionalBarcodeSpotRoi]))
            }
            times = rois.map(_.time).toSet
            numGroups <- Gen.choose(1, times.size)
            legitGroup <- Gen.oneOf(collections.partition(numGroups, times))
            repeated <- Gen.nonEmptyListOf(Gen.oneOf(times)).map(_.toSet)
            grouping = NonEmptyList(repeated, legitGroup).map(_.unsafeToNes)
            threshold <- arbitrary[PositiveReal]
            proxFilterStrategy <- Gen.oneOf(
                ImagingRoundsConfiguration.SelectiveProximityPermission(threshold, grouping),
                ImagingRoundsConfiguration.SelectiveProximityProhibition(threshold, grouping),
                )
        } yield (rois, repeated.unsafeToNes, proxFilterStrategy)
        
        // TODO: need to remove the pendingUntilFixed wrapper once this test is updated.
        // Originally, it was supposed to be the call under test which handled the prohibition on grouping components being mutually disjoint.
        // Now, this is instead handled at the level of the ImagingRoundsConfiguration.
        // The method under test has been made package-private, but in reality it should probably be made object-private, 
        // for exactly this region that the logic of restricting the structure of the regional imaging round grouping 
        // is expected to have already been done. Such a change would, however, require a further overhaul of some of these tests.
        // See: https://github.com/gerlichlab/looptrace/issues/266
        pendingUntilFixed{
            forAll (genSmallRoisAndGrouping) { (rois, repeatedTimepoints, proxFilterStrategy) =>
                buildNeighboringRoisFinder(NonnegativeInt.indexed(rois), proxFilterStrategy) match {
                    case Left(obsErrMsg) => 
                        val repTimesText = repeatedTimepoints.toList.map(t => t.get -> 2).sortBy(_._1).mkString(", ")
                        val expErrMsg = s"${repeatedTimepoints.size} repeated timepoint(s): $repTimesText"
                        obsErrMsg shouldEqual expErrMsg
                    case Right(_) => fail("Expected error message about invalid partition (non-disjoint), but didn't get it.")
                }
            }
        }
    }

    test("For UNIVERSAL PROHIBITION grouping, more proximal neighbors from different FOVs don't pair, while less proximal ones from the same FOV do pair. #150") {
        // The ROIs data for each test iteration
        val spotsText = """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
            |0,P0001.zarr,27,0,12,104,1000,4,20,100,108,996,1004
            |1,P0002.zarr,27,0,11,108,1002,3,21,100,116,998,1006
            |2,P0002.zarr,28,0,10,98,999,2,18,100,120,990,1008
            |3,P0001.zarr,28,0,11,99,998,3,19,99,108,988,1008
            |4,P0001.zarr,29,0,13,101,1004,5,21,90,112,996,1012
            |5,P0002.zarr,29,0,12,102,1003,1,23,91,113,995,1011
            |"""
        
        val allZeroDrift = """,frame,position,z_px_coarse,y_px_coarse,x_px_coarse,z_px_fine,y_px_fine,x_px_fine
            |0,27,P0001.zarr,0,0,0,0,0,0
            |1,28,P0001.zarr,0,0,0,0,0,0
            |2,29,P0001.zarr,0.0,0,0,0,0,0
            |3,27,P0002.zarr,0,0,0,0,0,0
            |4,28,P0002.zarr,0,0,0,0,0,0
            |5,29,P0002.zarr,0.0,0,0,0,0,0
            |"""

        forAll (arbitrary[ExtantOutputHandler]) { handleOutput => 
            withTempDirectory{ (tmpdir: os.Path) =>
                val spotsFile = tmpdir / "spots.csv"
                os.write(spotsFile, spotsText.stripMargin)
                val driftFile = tmpdir / "drift.csv"
                os.write(driftFile, allZeroDrift.stripMargin)
                val unfiltFile: UnfilteredOutputFile = UnfilteredOutputFile.fromPath(tmpdir / "unfiltered.csv")
                val filtFile: FilteredOutputFile = FilteredOutputFile.fromPath(tmpdir / "filtered.csv")
                workflow(
                    spotsFile = spotsFile, 
                    driftFile = driftFile, 
                    proximityFilterStrategy = ImagingRoundsConfiguration.UniversalProximityProhibition(PositiveReal(Double.PositiveInfinity)),
                    unfilteredOutputFile = unfiltFile, 
                    filteredOutputFile = filtFile, 
                    extantOutputHandler = handleOutput,
                    )
                safeReadAllWithOrderedHeaders(unfiltFile) match {
                    case Left(err) => throw err
                    case Right((_, rows)) => 
                        val observedNeighbors = rows.map(_("neighbors").split("\\|").toList.map(_.toInt).sorted)
                        val expectedNeighbors = List(List(3, 4), List(2, 5), List(1, 5), List(0, 4), List(0, 3), List(1, 2))
                        observedNeighbors shouldEqual expectedNeighbors
                }
            }
        }
    }

    test("For UNIVERSAL PROHIBITION grouping, ROI proximity labeling / neighbor finding is specific to field-of-view, so that ROIs from different FOVs don't affect each other for filtering. #150") {
        /* Create all partitions of 5 as 2 and 3, mapping each partition to a position value for ROIs to be made. */
        val roiIndices = 0 until 5
        val partitions = roiIndices.combinations(3).map{
            group => roiIndices.toList.map(i => PositionName(s"P000${if group.contains(i) then 1 else 2}.zarr"))
        }
        
        /* Assert property for all partitions (on position) of 5 ROIs into a group of 2 and group of 3. */
        forAll (Table("positions", partitions.toList*)) { positionAssigments =>
            /* Build ROIs with position assigned as randomised, and with index within list. */
            val roisWithIndex = NonnegativeInt.indexed(positionAssigments).map(_.leftMap{ pos => canonicalRoi.copy(position = pos) })
            /* Create the expected neighboring ROIs grouping, based on fact that all ROIs within each position group are coincident. */
            given orderForPair: Order[RoiLinenumPair] = Order.by(_._2)
            val expected = roisWithIndex.groupBy(_._1.position).values.map(_.map(_._2)).foldLeft(Map.empty[LineNumber, NonEmptySet[LineNumber]]){
                case (result, indexGroup) => 
                    val neighbors: NonEmptySet[LineNumber] = indexGroup.toNel.get.toNes
                    val subResult = indexGroup.map(i => i -> (neighbors - i).toNes.get).toMap
                    result |+| subResult
            }
            /* Only bother with 10 successes since threshold really should be irrelevant, and we're testing also over a table. */
            forAll (arbitrary[PositiveReal], minSuccessful(10)) { threshold => 
                buildNeighboringRoisFinder(roisWithIndex, ImagingRoundsConfiguration.UniversalProximityProhibition(threshold)) match {
                    case Left(errMsg) => fail(s"Expected test success but got failure/error message: $errMsg")
                    case Right(observed) => observed shouldEqual expected
                }
            }
        }
    }

    test("All-singleton PROHIBITIVE probe groupings guarantees no neighbors.") {
        def noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        def genRois: Gen[NonEmptyList[Roi]] = for {
            n <- Gen.choose(1, 10)
            regions <- Gen.pick(n, 0 until 100)
        } yield regions.map(r => canonicalRoi.copy(region = RegionId.unsafe(r))).toList.toNel.get
        
        forAll (genRois) { rois => 
            val proxFilterStrategy = ImagingRoundsConfiguration.SelectiveProximityProhibition(PositiveReal(1e-323), rois.map{ roi => NonEmptySet.one(roi.time) })
            buildNeighboringRoisFinder(NonnegativeInt.indexed(rois.toList), proxFilterStrategy) match {
                case Right(neigbors) => neigbors shouldEqual Map()
                case Left(errMsg) => fail(s"Expected success, but got error message: $errMsg")
            }
        }
    }

    test("Regardless of grouping semantic, fewer than 2 ROIs means the neighbors mapping is always empty.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        def genRoiDistanceAndRegionalImageRoundGroup: Gen[(RegionalBarcodeSpotRoi, ImagingRoundsConfiguration.NontrivialProximityFilter)] = {
            given arbMargin: Arbitrary[BoundingBox.Margin] = getArbForMargin(NonnegativeReal(1.0), NonnegativeReal(32.0))
            given arbPoint: Arbitrary[Point3D] = getArbForPoint3D(-2048.0, 2048.0)
            val genThreshold = arbitrary[PositiveReal]
            for {
                roi <- arbitrary[RegionalBarcodeSpotRoi]
                proxFilterStrategy <- {
                    val groups = NonEmptyList.one(NonEmptySet.one(roi.time))
                    Gen.oneOf(
                        genThreshold.flatMap{ t => Gen.oneOf(
                            ImagingRoundsConfiguration.SelectiveProximityPermission(t, groups),
                            ImagingRoundsConfiguration.SelectiveProximityProhibition(t, groups),
                        )},
                        genThreshold.map(ImagingRoundsConfiguration.UniversalProximityProhibition.apply),
                    )
                }
            } yield (roi, proxFilterStrategy)
        }
        
        forAll (genRoiDistanceAndRegionalImageRoundGroup) { (roi, proxFilterStrategy) => 
            val rois = NonnegativeInt.indexed(List(roi))
            buildNeighboringRoisFinder(rois, proxFilterStrategy) shouldEqual Right(Map())
        }
    }

    test("A ROI is never among its own neighbors.") {
        forAll (genThresholdAndRoisToFacilitateCollisions) { (threshold, rois) => 
            buildNeighboringRoisFinder(NonnegativeInt.indexed(rois), ImagingRoundsConfiguration.UniversalProximityProhibition(threshold)) match {
                case Left(errMsg) => fail(s"Expected success, but got error message: $errMsg")
                case Right(neighbors) => neighbors.toList.filter{ case (k, vs) => vs `contains` k } shouldEqual List()
            }
        }
    }

    test("Neighbor relation is bidirectional relation, so each of a ROI's neighbors has the ROI among its own neighbors.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        forAll (genThresholdAndRoisToFacilitateCollisions) { (threshold, rois) => 
            buildNeighboringRoisFinder(NonnegativeInt.indexed(rois), ImagingRoundsConfiguration.UniversalProximityProhibition(threshold)) match {
                case Left(errMsg) => fail(s"Expected success, but got error message: $errMsg")
                case Right(neighbors) => 
                    val (fails, _) = Alternative[List].separate(neighbors.toList.flatMap{ 
                        case curr@(k, vs) => vs.toList.map{ v => neighbors(v).contains(k).either(curr -> (v -> neighbors(v)), ()) }
                    })
                    fails shouldEqual List()
            }
        }
    }

    test("Zero threshold guarantees no neighbors.") {
        given arbitraryDoubleTemp: Arbitrary[Double] = genReasonableCoordinate.toArbitrary
        forAll (arbitrary[List[Point3D]], Gen.resize(5, Gen.alphaNumStr)) { (pts, key) => 
            val keyedPoints = pts.map(key -> _)
            val threshold = PiecewiseDistance.ConjunctiveThreshold(PositiveReal(1e-323).asNonnegative)
            buildNeighborsLookupKeyed(keyedPoints, (_, _) => true, identity[Point3D], threshold, identity) shouldEqual Map()
        }
    }

    test("Total uniqueness of keys guarantees no neighbors.") {
        forAll { (minDist: PositiveReal, points: List[Point3D]) => 
            val threshold = PiecewiseDistance.ConjunctiveThreshold(minDist.asNonnegative)
            buildNeighborsLookupKeyed(points.zipWithIndex.map(_.swap), (_, _) => true, identity[Point3D], threshold, identity) shouldEqual Map()
        }
    }

    test("ROIs from different channels can never exclude one another. #138") {
        // TODO: implement with multi-channel adaptations for IF.
        // See: https://github.com/gerlichlab/looptrace/issues/138
        pending
    }
    
    /****************************************************************************************************************
     * Ancillary definitions
     ****************************************************************************************************************/
    private def canonicalRoi: Roi = canonicalRoi(Point3D(XCoordinate(1), YCoordinate(2), ZCoordinate(3)))
    private def canonicalRoi(point: Point3D): Roi = {
        point match { case Point3D(x, y, z) => 
            val xIntv = buildInterval(x, BoundingBox.Margin(NonnegativeReal(2)))(XCoordinate.apply)
            val yIntv = buildInterval(y, BoundingBox.Margin(NonnegativeReal(2)))(YCoordinate.apply)
            val zIntv = buildInterval(z, BoundingBox.Margin(NonnegativeReal(1)))(ZCoordinate.apply)
            val box = BoundingBox(xIntv, yIntv, zIntv)
            RegionalBarcodeSpotRoi(
                RoiIndex(NonnegativeInt(0)), 
                PositionName("P0001.zarr"), 
                RegionId(Timepoint(NonnegativeInt(0))), 
                Channel(NonnegativeInt(0)), 
                point, 
                box,
                )
        }
    }

    def genSpotCoveringGrouping(genThreshold: Gen[PositiveReal])(spots: Iterable[RegionalBarcodeSpotRoi]) = {
        given ordTime: Ordering[Timepoint] = Order[Timepoint].toOrdering
        val timepoints = spots.map(_.region.get).toSet
        for {
            maybeSplit <- Gen.option(Gen.choose(1, timepoints.size - 1))
            groups = maybeSplit match {
                case None => NonEmptyList.one(timepoints.toNonEmptySetUnsafe)
                case Some(k) => 
                    val (g1, g2) = timepoints.toList.splitAt(k)
                    NonEmptyList.of(g1, g2).map(_.toSet.toNonEmptySetUnsafe)
            }
            threshold <- genThreshold
            grouping <- Gen.oneOf(
                ImagingRoundsConfiguration.SelectiveProximityPermission(threshold, groups),
                ImagingRoundsConfiguration.SelectiveProximityProhibition(threshold, groups),
                )
        } yield grouping
    }

    /** Use the given drift component generators to create a full drift record for each generated spot record. */
    private def genSpotsAndDrifts(genCoarse: Gen[CoarseDrift], genFine: Gen[FineDrift])(
        using arbMargin: Arbitrary[BoundingBox.Margin], arbPoint: Arbitrary[Point3D]
        ): Gen[(NonEmptyList[RegionalBarcodeSpotRoi], List[(PositionName, Timepoint, CoarseDrift, FineDrift)])] = {
        // Order shouldn't matter, but that invariant's tested elsewhere.
        given ordPosTime: Ordering[DriftKey] = Order[DriftKey].toOrdering
        for {
            spots <- Gen.nonEmptyListOf(arbitrary[RegionalBarcodeSpotRoi]).map(_.toNel.get)
            posTimePairs = spots.toList.map(roi => roi.position -> roi.time).toSet
            driftRows <- posTimePairs.toList.traverse{ 
                (p, t) => Gen.zip(genCoarse, genFine).map((coarse, fine) => (p, t, coarse, fine))
            }
        } yield (spots, driftRows.sortBy(r => r._1 -> r._2))
    }

    private def genSpotsAndDriftsWithDrop(genCoarse: Gen[CoarseDrift], genFine: Gen[FineDrift])(
        using arbMargin: Arbitrary[BoundingBox.Margin], arbPoint: Arbitrary[Point3D]
        ): Gen[(NonEmptyList[RegionalBarcodeSpotRoi], List[(PositionName, Timepoint, CoarseDrift, FineDrift)], Int)] = {
        // Order shouldn't matter, but that invariant's tested elsewhere.
        given ordPosTime: Ordering[DriftKey] = Order[DriftKey].toOrdering
        genSpotsAndDrifts(genCoarse, genFine).flatMap{ (spots, driftRows) => 
            Gen.choose(0, driftRows.size).map{ numDropped => 
                // Shuffle and the re-sort so that removed rows aren't always from the beginning.
                val rows = Random.shuffle(driftRows).toList.drop(numDropped)
                (spots, rows.sortBy(r => r._1 -> r._2), numDropped)
            }
        }
    }
    
    private def buildInterval[C <: Coordinate: [C] =>> NotGiven[C =:= Coordinate]](
        c: C, margin: BoundingBox.Margin)(lift: Double => C): BoundingBox.Interval[C] = 
        BoundingBox.Interval[C].apply.tupled((c.get - margin.get, c.get + margin.get).mapBoth(lift))

    private def genThresholdAndRoisToFacilitateCollisions: Gen[(PositiveReal, List[Roi])] = {
        for {
            threshold <- Gen.choose(1e-323, 10.0).map(PositiveReal.unsafe)
            numRois <- Gen.choose(5, 10)
            centroids <- {
                given tmpArb: Arbitrary[Double] = Arbitrary(genNonNegReal(NonnegativeReal(5)))
                Gen.listOfN(numRois, arbitrary[Point3D])
            }
        } yield (threshold, centroids.map(canonicalRoi))
    }

    private def getArbForMargin(lo: NonnegativeReal, hi: NonnegativeReal): Arbitrary[BoundingBox.Margin] = 
        Gen.choose[Double](lo, hi).map(NonnegativeReal.unsafe `andThen` BoundingBox.Margin.apply).toArbitrary

    private def getArbForPoint3D(lo: Double, hi: Double): Arbitrary[Point3D] = arbitraryForPoint3D(using Gen.choose(lo, hi).toArbitrary)

    private def getDriftFileLines(driftRows: List[(PositionName, Timepoint, CoarseDrift, FineDrift)]): List[String] = 
        headDriftFile :: driftRows.zipWithIndex.map{ case ((pos, time, coarse, fine), i) => 
            s"$i,${time.get},${pos.get},${coarse.z.get},${coarse.y.get},${coarse.x.get},${fine.z.get},${fine.y.get},${fine.x.get}"
        }

    private def getSpotsFileLines(rois: List[RegionalBarcodeSpotRoi]): List[String] = headSpotsFile :: rois.map{ r => 
        val (x, y, z) = r.centroid match { case Point3D(x, y, z) => (x.get, y.get, z.get) }
        val (loX, hiX, loY, hiY, loZ, hiZ) = r.boundingBox match { 
            case BoundingBox(sideX, sideY, sideZ) => (sideX.lo.get, sideX.hi.get, sideY.lo.get, sideY.hi.get, sideZ.lo.get, sideZ.hi.get)
        }
        s"${r.index.get},${r.position.get},${r.time.get},${r.channel.get},$x,$y,$z,$loX,$hiX,$loY,$hiY,$loZ,$hiZ"
    }

    // Header for spots (ROIs) file
    private def headSpotsFile = ",position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max"

    // Header for drift correction file
    private def headDriftFile = ",frame,position,z_px_coarse,y_px_coarse,x_px_coarse,z_px_fine,y_px_fine,x_px_fine"

end TestLabelAndFilterRois
