package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Random
import cats.*
import cats.syntax.all.*
import upickle.default.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.testing.syntax.SyntaxForScalacheck

import at.ac.oeaw.imba.gerlich.looptrace.TracingOutputAnalysis.*
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.readJsonFile
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Tests for pair of regional and locus-specific spot image timepoints. */
class TestSpotTimePair extends AnyFunSuite, ScalaCheckPropertyChecks, LooptraceSuite, SyntaxForScalacheck, should.Matchers:
    import SpotTimePair.given
    
    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)

    test("SpotTimePair equivalence is component-wise.") {
        /* Confine the bounds to generate more collisions */
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        given arbNonNegInt: Arbitrary[NonnegativeInt] = 
            Arbitrary{ Gen.choose(1, 10).map(NonnegativeInt.unsafe) }
        
        // Bump up min success count to more thoroughly explore the space.
        forAll (minSuccessful(10000)) { (pair1: SpotTimePair, pair2: SpotTimePair) =>
            val byInstance = Eq[SpotTimePair].eqv(pair1, pair2)
            val byComponent = (pair1, pair2) match {
                case (
                    (RegionId(rt1), LocusId(lt1)), 
                    (RegionId(rt2), LocusId(lt2))
                ) => rt1 === rt2 && lt1 === lt2
            }
            byInstance shouldBe byComponent
        }
    }
    
    test("SpotTimePair roundtrips through JSON.") {
        forAll { (pair: SpotTimePair) => pair shouldEqual read[SpotTimePair](write(pair)) }
    }

    test("Accurate region-locus timepoint filter can parse from JSON (raw or file) or CSV (file).") {
        import RegLocFilter.given
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        
        val roundtripsTable = Table(
            ("roundtrip", "extension"), 
            ThroughJsonPure[RegLocFilter]() -> "whatever", 
            ThroughJsonFile[RegLocFilter]() -> "json", 
            ThroughCsvFile -> "csv",
            )
        
        forAll (genPairsAndExtra) { (pairsToUse, extraPairs) => 
            forAll (roundtripsTable) { (roundtrip, extension) =>
                withTempDirectory{ (tmpdir: os.Path) => 
                    val pairsFile = tmpdir / s"pairs_to_use.$extension"
                    val includePair = roundtrip(pairsToUse.toList, pairsFile)
                    val allPairs = pairsToUse ++ extraPairs
                    val expected = if (pairsToUse.isEmpty) then allPairs else pairsToUse
                    val observed = Random.shuffle(allPairs).filter(includePair)
                    observed.toSet shouldEqual expected
                }
            }
        }
    }

    test("Repeats in pairs of interest is an error through JSON (raw or file) or CSV (file).") {
        import RegLocFilter.given
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        
        val roundtripsTable = Table(
            ("roundtrip", "extension"), 
            ThroughJsonPure[RegLocFilter]() -> "whatever", 
            ThroughJsonFile[RegLocFilter]() -> "json", 
            ThroughCsvFile -> "csv",
            )

        forAll (genPairsWithCollision) { pairs =>
            forAll (roundtripsTable) { (roundtrip, extension) => 
                withTempDirectory { (tmpdir: os.Path) => 
                    val pairsFile = tmpdir / s"pairs.$extension"
                    assertThrows[RegLocFilter.NonUniquenessException]{ roundtrip(pairs, pairsFile) }
                }
            }
        }
    }

    test("Counts by spot time pair are accurate, declaring filter as empty or JSON or CSV.") {
        /* The hypothetical lines in which to count records by (ref_frame, frame) */
        val lines = List(
            "position,pos_id,ref_frame,roi_id,frame,z,y,x",
            "P0001.zarr,0,5,0,0,-1.0,2.0,1.0",
            "P0001.zarr,0,5,0,1,0.0,3.0,4.0",
            "P0001.zarr,0,5,0,2,-2.0,1.0,0.0",
            "P0001.zarr,0,6,0,0,-3.0,4.0,2.0",
            "P0001.zarr,0,6,0,0,-4.0,5.0,3.0",
            "P0001.zarr,0,6,0,1,1.0,6.0,5.0",
            "P0002.zarr,1,5,0,0,2.0,-1.0,0.0",
            "P0002.zarr,1,5,0,1,3.0,-2.0,-10.0",
            "P0002.zarr,1,5,0,1,4.0,-3.0,-8.0",
            "P0002.zarr,1,7,0,2,5.0,-4.0,-6.0",
            "P0002.zarr,1,7,0,2,6.0,-5.0,-4.0",
            "P0002.zarr,1,7,0,2,7.0,-6.0,-2.0",
        )

        /* Create expected result over powerset of potential inputs, given above data. */
        val allExpectations = {
            // First, map each possible query pair to its record count.
            val expIfAllUsed = List(
                (5, 0) -> 2, (5, 1) -> 3, (5, 2) -> 1, 
                (6, 0) -> 2, (6, 1) -> 1,
                (7, 2) -> 3
                ).map(_.leftMap(_.mapBoth(NonnegativeInt.unsafe).toSpotTimePair)).toMap
            // Then, map each element of the powerset over keys to corresponding subset of results.
            val expectations = (1 to expIfAllUsed.size).toList.flatMap{ n => 
                Functor[List].fproduct(
                    expIfAllUsed.keySet.toList.combinations(n).toList
                    )(combos => expIfAllUsed.view.filterKeys(combos.contains).toMap)
            }
            // Next, allow each possible input to be written by JSON or CSV.
            val augmentedExpectations = for {
                (pairs, expect) <- ((Nil, expIfAllUsed) :: expectations)
                (func, ext) <- List(
                    (writeToCsv(pairs, _: os.Path), "csv"), 
                    (os.write(_: os.Path, write(pairs, indent = 2)), "json"),
                    )
            } yield ((func, ext), expect)
            // Finally, consider the null possibility for input.
            (None, expIfAllUsed) :: augmentedExpectations.map(_.leftMap(_.some))
        }
        
        forAll (Table(("maybeWriteAndExt", "expected"), allExpectations*)) { 
            (maybeWriteAndExt, expected) => 
                withTempDirectory{ (tmpdir: os.Path) => 
                    val dataFile = tmpdir / "data.csv"
                    os.write(dataFile, lines.map(_ ++ "\n"))
                    val observed = maybeWriteAndExt match {
                        case None => countByRegionLocusPairUnsafe(dataFile)
                        case Some((writePairs, ext)) => 
                            val pairsFile = tmpdir / s"pairs_to_use.$ext"
                            writePairs(pairsFile)
                            countByRegionLocusPairUnsafe(pairsOfInterestFile = pairsFile, dataFile = dataFile)
                    }
                    observed shouldEqual expected
                }
        }
    }

    test("Any filepath matching in call to write counts is an error.") {
        val writerFilenamesPairs = {
            val runUnfiltered = { (pairsFile: os.Path, infile: os.Path, outfile: os.Path) => 
                writeRegionalLocalPairCountsFiltered(pairsFile)(infile, outfile)
            }
            val unfiltered = (
                { (_: os.Path, infile: os.Path, outfile: os.Path) =>
                    writeRegionalLocalPairCounts(infile, outfile)
                }, 
                ("irrelevant.csv", "pairs.csv", "pairs.csv")
                )
            val filtered = List(
                ("pairs.csv", "pairs.csv", "out.csv"), 
                ("pairs.csv", "infile.csv", "pairs.csv"), 
                ("pairs.csv", "infile.csv", "infile.csv"), 
                ("out.csv", "out.csv", "out.csv"),
                ).map(runUnfiltered -> _)
            unfiltered :: filtered
        }

        forAll (Table(("writeCounts", "filenames"), writerFilenamesPairs*)) { (writeCounts, filenames) => 
            withTempDirectory{ (tmpdir: os.Path) =>
                val (pairsFn, inFn, outFn) = filenames
                val pairsFile = tmpdir / pairsFn
                val infile = tmpdir / inFn
                val outfile = tmpdir / outFn
                assertThrows[IllegalArgumentException]{ writeCounts(pairsFile, infile, outfile) }
            }
        }
    }

    test("Attempt to count grouped records with an interest file is error if interest file has unknown extension.") {
        val lines = List(
            "position,pos_id,ref_frame,roi_id,frame,z,y,x",
            "P0001.zarr,0,5,0,0,-1.0,2.0,1.0",
            "P0001.zarr,0,5,0,1,0.0,3.0,4.0",
            "P0001.zarr,0,5,0,2,-2.0,1.0,0.0",
            "P0001.zarr,0,6,0,0,-3.0,4.0,2.0",
            "P0001.zarr,0,6,0,0,-4.0,5.0,3.0",
        )
        val writers = {
            val pairs = List(5 -> 1, 6 -> 0).map(_.mapBoth(NonnegativeInt.unsafe).toSpotTimePair)
            List(
                os.write(_, write(pairs, indent = 2)), 
                writeToCsv(pairs, _),
            )
        }

        forAll (Table("writePairs", writers*)) { writePairs => 
            forAll (Gen.alphaNumStr.suchThat(ext => ext =!= "csv" && ext =!= "json")) { ext =>
                withTempDirectory{ (tmpdir: os.Path) =>
                    val pairsFile = tmpdir / s"pairs.$ext"
                    writePairs(pairsFile)
                    val infile = tmpdir / "records.csv"
                    os.write(infile, lines.map(_ ++ "\n"))
                    val outfile = tmpdir / "out.csv"
                    val caught = intercept[IllegalArgumentException]{
                        writeRegionalLocalPairCountsFiltered(pairsFile)(infile, outfile)
                    }
                    val expMsg = s"Cannot parse pairs of interest file with extension '$ext': $pairsFile"
                    caught.getMessage shouldEqual expMsg
                }
            }
        }
    }

    test("Mismatch--when writing counts--between extension and delimiter is an error, whether filtering or not.") {
        val lines = List(
            "position,pos_id,ref_frame,roi_id,frame,z,y,x",
            "P0001.zarr,0,5,0,0,-1.0,2.0,1.0",
            "P0001.zarr,0,5,0,1,0.0,3.0,4.0",
            "P0001.zarr,0,5,0,2,-2.0,1.0,0.0",
            "P0001.zarr,0,6,0,0,-3.0,4.0,2.0",
            "P0001.zarr,0,6,0,0,-4.0,5.0,3.0",
        )
        val writerExtMaybes = {
            val pairs = List(5 -> 1, 6 -> 0).map(_.mapBoth(NonnegativeInt.unsafe).toSpotTimePair)
            List(
                Option{ (os.write(_, write(pairs, indent = 2)), "json") }, 
                Option{ (writeToCsv(pairs, _), "csv") },
                None
            )
        }

        forAll (Table("maybeWriteAndExt", writerExtMaybes*)) { maybeWriteAndExt => 
            forAll (Gen.alphaNumStr.suchThat(_ =!= "csv")) { outFileExt => 
                withTempDirectory{ (tmpdir: os.Path) =>
                    val callFunc = maybeWriteAndExt match {
                        case None => writeRegionalLocalPairCounts
                        case Some((writePairs, pairsFileExt)) => 
                            val pairsFile = tmpdir / s"pairs.$pairsFileExt"
                            writePairs(pairsFile)
                            writeRegionalLocalPairCountsFiltered(pairsFile)
                    }
                    val infile = tmpdir / "records.csv"
                    os.write(infile, lines.map(_ ++ "\n"))
                    val outfile = tmpdir / s"output.$outFileExt"
                    val caught = intercept[IllegalArgumentException]{ callFunc(infile, outfile) }
                    val expMsg = s"Unexpected extension ('${outfile.ext}', not 'csv') for output file: $outfile"
                    caught.getMessage shouldEqual expMsg
                }
            }
        }
    }

    // integration-like test for a simple use case, no filtering
    test("Writing counts to CSV works.") {
        /* Input and expected output data */
        val (allInlines, expOutLines) = {
            val lineByPair = List(
                (6 -> 0) -> "P0001.zarr,0,6,0,0,-3.0,4.0,2.0", 
                (6 -> 0) -> "P0001.zarr,0,6,0,0,-4.0,5.0,3.0",
                (5 -> 1) -> "P0001.zarr,0,5,0,1,0.0,3.0,4.0",
                (5 -> 0) -> "P0001.zarr,0,5,0,0,-1.0,2.0,1.0",
                (5 -> 1) -> "P0001.zarr,0,5,0,1,-2.0,1.0,0.0",
                )
            val allIn = lineByPair.map(_._2)
                .permutations
                .toList
                .map("position,pos_id,ref_frame,roi_id,frame,z,y,x" :: _)
            val expOut = {
                val expData = lineByPair.groupBy(_._1).view.mapValues(_.length).toList
                "regional,local,N" :: expData.sortBy(_._1).map{ case ((reg, loc), n) => s"$reg,$loc,$n" }
            }
            allIn -> expOut
        }

        forAll (Table("inlines", allInlines*)) { inlines => 
            withTempDirectory{ (tmpdir: os.Path) =>
                val infile = tmpdir / "in.csv"
                os.write(infile, inlines.map(_ ++ "\n"))
                val outfile = tmpdir / "out.csv"
                writeRegionalLocalPairCounts(infile, outfile)
                val obsOutLines = os.read.lines(outfile).toList
                obsOutLines shouldEqual expOutLines
            }
        }
    }

    /* Helpers for constructing inputs */
    extension (n: NonnegativeInt)
        def toLocal = LocusId(ImagingTimepoint(n))
        def toRegional = RegionId(ImagingTimepoint(n))
    extension (nn: (NonnegativeInt, NonnegativeInt))
        def toSpotTimePair: SpotTimePair = nn.bimap(_.toRegional, _.toLocal)

    private sealed trait Roundtrip[A] extends Function2[List[SpotTimePair], os.Path, A]
    private final case class ThroughJsonPure[A : Reader]() extends Roundtrip[A]:
        def apply(pairs: List[SpotTimePair], f: os.Path) = read(write(pairs, indent = 2))
    private final case class ThroughJsonFile[A : Reader]() extends Roundtrip[A]:
        def apply(pairs: List[SpotTimePair], f: os.Path) = 
            os.write(f, write(pairs, indent = 2))
            readJsonFile(f)
    private case object ThroughCsvFile extends Roundtrip[RegLocFilter]:
        def apply(pairs: List[SpotTimePair], f: os.Path) =
            writeToCsv(pairs, f)
            RegLocFilter.fromCsvFileUnsafe(f)
    
    private final def genPairsAndExtra: Gen[(Set[SpotTimePair], Set[SpotTimePair])] = for {
        allPairs <- arbitrary[Set[SpotTimePair]].suchThat(_.nonEmpty)
        toUse <- Gen.choose(0, allPairs.size).flatMap(Gen.pick(_, allPairs)).map(_.toSet)
    } yield (toUse, allPairs -- toUse)

    private final def writeToCsv(pairs: List[SpotTimePair], f: os.Path) = 
        val textPairs = pairs.map(_.bimap(_.show_, _.show_))
        val lines = (("ref_frame", "frame") :: textPairs).map((_1, _2) => s"${_1},${_2}\n")
        os.write(f, lines)

    private final def genPairsWithCollision = {
        val gen = for {
            n <- Gen.choose(5, 10)
            pairs <- {
                given arbNonNeg: Arbitrary[NonnegativeInt] = 
                    Arbitrary{ Gen.choose(0, 2).map(NonnegativeInt.unsafe) }
                Gen.listOfN(n, arbitrary[SpotTimePair])
            }
        } yield pairs
        gen.suchThat{ pairs => pairs.toSet.size < pairs.length }
    }

end TestSpotTimePair
