package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Random
import cats.syntax.eq.*
import cats.syntax.flatMap.*
import upickle.default.*

import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.looptrace.PartitionIndexedPoints.{
    BeadRoisPrefix, 
    ParserConfig, 
    XColumn, 
    YColumn, 
    ZColumn, 
    discoverInputs, 
    getOutputFilename, 
    sampleDetectedRois
}
import at.ac.oeaw.imba.gerlich.looptrace.space.CoordinateSequence

/** Tests for the partitioning of regions of interest (ROIs) for drift correction */
class TestPartitionIndexedDriftCorrectionRois extends AnyFunSuite with ScalaCheckPropertyChecks with should.Matchers with PartitionRoisSuite {
    import SelectedRoi.*
    
    final case class Dataset(lines: Iterable[String], expectedNumTotal: PositiveInt, expectedNumDiscarded: PositiveInt) {
        final def expectedNumAvailable: Int = expectedNumTotal - expectedNumDiscarded
    }

    trait DatasetLike[-D]:
        def getDataset: D => Dataset

    object DatasetLike:
        given datasetLikeForTuple2: DatasetLike[(Any, Dataset)] with
            def getDataset: ((Any, Dataset)) => Dataset = _._2

    def genDataset: Gen[Dataset] = Gen.oneOf(dataset1, dataset2)

    def getInputFilename(pos: PositionIndex, frame: FrameIndex): String = s"bead_rois__${pos.get}_${frame.get}.csv"

    test("Requesting sample size greater than available record count is an error.") {
        val maxRoisCount = 100
        
        def genSampleSizeInExcessOfAllRois: Gen[(PositiveInt, PositiveInt, Iterable[DetectedRoi])] = for {
            rois <- Gen.choose(0, maxRoisCount).flatMap{ n => Gen.listOfN(n, arbitrary[DetectedRoi]) }
            del <- Gen.choose(1, maxRoisCount).map(PositiveInt.unsafe)
            acc <- Gen.choose(scala.math.max(0, rois.size - del) + 1, maxRoisCount).map(PositiveInt.unsafe)
        } yield (del, acc, rois)

        def genSampleSizeInExcessOfAvailableRois: Gen[(PositiveInt, PositiveInt, Iterable[DetectedRoi])] = for {
            rois <- Gen.choose(2, maxRoisCount).flatMap{ n => Gen.listOfN(n, arbitrary[DetectedRoi]).suchThat(_.exists(_.isUsable)) }
            numUsable = rois.count(_.isUsable)
            del <- Gen.choose(1, numUsable).map(PositiveInt.unsafe)
            acc <- Gen.choose(numUsable - del + 1, rois.size - del).map(PositiveInt.unsafe)
        } yield (del, acc, rois)
        
        forAll (Gen.oneOf(genSampleSizeInExcessOfAllRois, genSampleSizeInExcessOfAvailableRois)) { 
            case (numShifting, numAccuracy, rois) => 
                assert(numShifting + numAccuracy > rois.count(_.isUsable), "Uh-oh!")
                val observed = sampleDetectedRois(numShifting, numAccuracy)(rois)
                val expected = {
                    val sampleSize = numShifting + numAccuracy
                    Left(s"Fewer ROIs available than requested! ${rois.count(_.isUsable)} < ${sampleSize}")
                }
                observed shouldBe expected
        }
    }

    test("Input discovery works as expected for folder with no other contents.") {
        type NNPair = (NonnegativeInt, NonnegativeInt)
        type PosFramePair = (PositionIndex, FrameIndex)
        def genNonnegativePair: Gen[NNPair] = Gen.zip(genNonnegativeInt, genNonnegativeInt)
        def genDistinctNonnegativePairs: Gen[(PosFramePair, PosFramePair)] = 
            Gen.zip(genNonnegativePair, genNonnegativePair)
                .suchThat{ case (p1, p2) => p1 =!= p2 }
                .map { case ((p1, f1), (p2, f2)) => (PositionIndex(p1) -> FrameIndex(f1), PositionIndex(p2) -> FrameIndex(f2)) }
        forAll (genDistinctNonnegativePairs) { case (pf1, pf2) => {
            withTempDirectory{ (p: os.Path) => 
                val expected = List(pf1, pf2) map { case (pos, frame) => (pos, frame, p / getInputFilename(pos, frame)) }

                /* Check that inputs don't already exist, then establish them and check existence. */
                expected.exists{ case (_, _, fp) => os.exists(fp) } shouldBe false
                expected.foreach{ case (_, _, fp) => touchFile(fp) }
                expected.forall{ case (_, _, fp) => os.isFile(fp) } shouldBe true

                val found = discoverInputs(p) // Perform the empirical action.
                found.length shouldEqual expected.length // Check size before reducing to Set.
                found.toSet shouldEqual expected.toSet // Ignore ordering.
            }
        } }
    }

    test("Input discovery works as expected for mixed folder contents.") { (pending) }

    test("Sampling yields collections among which indices have no overlap.") { (pending) }

    test("Simple golden path") {
        (pending)
    }

    test("Collision between column-like names in parser config is illegal") {
        /* Build the random input generator for this test, assuring that at least one column name collision occurs */
        val uniqueValueCounts = List(List(1, 1, 2), List(2, 2), List(1, 3), List(4))
        def genElems: List[Int] => Gen[List[String]] = structure => {
            val genBlock: Gen[List[List[String]]] = 
                Gen.sequence(structure map { n => arbitrary[String].map(List.fill(n)) })
            genBlock.map(_.flatten)
        }
        def genTextLikes: Gen[(XColumn, YColumn, ZColumn, String)] = {
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
        def genBadConfigInputs: Gen[(XColumn, YColumn, ZColumn, String, CoordinateSequence)] = for {
            (x, y, z, qc) <- genTextLikes
            cs <- arbitrary[CoordinateSequence]
        } yield (x, y, z, qc, cs)
        
        forAll(genBadConfigInputs) { tup => assertThrows[IllegalArgumentException]{ ParserConfig.apply.tupled(tup) } }
    }

    test("Parser config roundtrips through JSON") {
        forAll { (original: ParserConfig) => 
            val jsonData = write(original, indent = 2)
            withTempFile(jsonData){ readJson[ParserConfig](_: os.Path) shouldEqual original }
        }
    }

    def standardConfigLines = """
    {
        "xCol": "centroid-0",
        "yCol": "centroid-1",
        "zCol": "centroid-2",
        "qcCol": "fail_code",
        "coordinateSequence": "Reverse"
    }
    """.split("\n")

    def dataset1 = Dataset(lines1, expectedNumTotal = PositiveInt(7), expectedNumDiscarded = PositiveInt(3))
    def lines1 = """,label,centroid-0,centroid-1,centroid-2,max_intensity,area,fail_code
        101,102,11.96875,1857.9375,1076.25,26799.0,32.0,
        104,105,10.6,1919.8,1137.4,12858.0,5.0,i
        109,110,11.88,1939.52,287.36,21065.0,25.0,
        110,111,11.5,1942.0,1740.625,21344.0,32.0,
        115,116,11.35,2031.0,863.15,19610.0,20.0,
        116,117,12.4,6.4,1151.5,16028.0,10.0,y
        117,118,12.1,8.1,1709.5,14943.0,10.0,i
        """.split("\n").map(_.trim)

    def dataset2 = Dataset(lines2, expectedNumTotal = PositiveInt(8), expectedNumDiscarded = PositiveInt(2))
    def lines2 = """,label,centroid-0,centroid-1,centroid-2,max_intensity,area,fail_code
        3,4,9.6875,888.375,1132.03125,25723.0,32.0,
        20,21,10.16,1390.94,1386.96,33209.0,50.0,
        34,35,9.3125,1567.5,87.40625,23076.0,32.0,
        36,37,9.166666666666666,1576.75,18.0,16045.0,12.0,
        41,42,9.0,1725.4,1886.8,12887.0,5.0,i
        43,44,9.0,1745.6,1926.1,15246.0,10.0,
        44,47,8.875,1851.25,1779.5,14196.0,8.0,i
        46,45,9.708333333333334,1831.625,1328.0,22047.0,24.0,
        """.split("\n").map(_.trim)
}
