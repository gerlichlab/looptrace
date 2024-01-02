package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import cats.*
import cats.data.NonEmptySet
import cats.syntax.all.*
import mouse.boolean.*

import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*

import at.ac.oeaw.imba.gerlich.looptrace.space.{ 
    Coordinate, 
    DistanceThreshold,
    EuclideanDistance, 
    PiecewiseDistance, 
    Point3D, 
    XCoordinate, 
    YCoordinate, 
    ZCoordinate
}
import at.ac.oeaw.imba.gerlich.looptrace.LabelAndFilterRois.{
    buildNeighborsLookupFlat, 
    buildNeighborsLookupKeyed, 
    buildNeighboringRoisFinder, 
    workflow as runLabelAndFilter,
    BoundingBox, 
    FilteredOutputFile,
    LineNumber, 
    ProbeGroup,
    Roi, 
    RoiLinenumPair, 
    UnfilteredOutputFile,
}


/** Tests for the filtration of the individual supports (single FISH probes) of chromatin fiber traces */
class TestLabelAndFilterRois extends AnyFunSuite, DistanceSuite, LooptraceSuite, ScalacheckSuite, should.Matchers:

    test("Drift file is required.") {
        def genThresholdAndHandler = for {
            threshold <- genThreshold(arbitrary[NonnegativeReal])
            extantHandler <- arbitrary[ExtantOutputHandler]
        } yield (threshold, extantHandler)
        forAll (genThresholdAndHandler) { (threshold, extantHandler) => 
            withTempDirectory{ (tmpdir: os.Path) => 
                // missing driftFile
                assertTypeError{ "runLabelAndFilter( " +
                    "spotsFile = tmpdir / \"traces.csv\", " + 
                    "probeGroups = List(), " + 
                    "minSpotSeparation = PiecewiseDistance.ConjunctiveThreshold(NonnegativeReal(5.0)), " + 
                    "unfilteredOutputFile = UnfilteredOutputFile.fromPath(tmpdir / \"unfiltered.csv\"), " + 
                    "filteredOutputFile = FilteredOutputFile.fromPath(tmpdir / \"filtered.csv\"), " + 
                    "extantOutputHandler = extantHandler " + 
                    ")"
                }
                // Add in driftFile to get compilation.
                assertCompiles{ "runLabelAndFilter( " +
                    "spotsFile = tmpdir / \"traces.csv\", " + 
                    "driftFile = tmpdir / \"drift.csv\", " + 
                    "probeGroups = List(), " + 
                    "minSpotSeparation = PiecewiseDistance.ConjunctiveThreshold(NonnegativeReal(5.0)), " + 
                    "unfilteredOutputFile = UnfilteredOutputFile.fromPath(tmpdir / \"unfiltered.csv\"), " + 
                    "filteredOutputFile = FilteredOutputFile.fromPath(tmpdir / \"filtered.csv\"), " + 
                    "extantOutputHandler = extantHandler " + 
                    ")"
                }
            }
        }
    }

    test("Spot distance comparison uses drift correction.") { pending }
    
    val spotsLines = """,position,frame,ch,zc,yc,xc,z_min,z_max,y_min,y_max,x_min,x_max
        |0,P0001.zarr,27,0,18.594063700840934,104.97590586866923,1052.9315138200425,10.594063700840934,26.594063700840934,88.97590586866923,120.97590586866923,1036.9315138200425,1068.9315138200425
        |1,P0001.zarr,27,0,18.45511019130035,1739.9764501391553,264.9779910476261,10.45511019130035,26.45511019130035,1723.9764501391553,1755.9764501391553,248.97799104762612,280.9779910476261
        |2,P0001.zarr,27,0,3.448146886449433,1878.0134803495575,31.018454661114163,-4.5518531135505675,11.448146886449432,1862.0134803495575,1894.0134803495575,15.018454661114163,47.01845466111416
        |3,P0001.zarr,27,0,6.639437282025158,1380.9842529095968,1457.0201320584663,-1.360562717974842,14.639437282025158,1364.9842529095968,1396.9842529095968,1441.0201320584663,1473.0201320584663
        |4,P0001.zarr,27,0,7.366868362813202,24.020457573537264,1359.9764542079788,-0.6331316371867981,15.366868362813202,8.020457573537264,40.020457573537264,1343.9764542079788,1375.9764542079788
        |5,P0001.zarr,27,0,10.747546275311462,1783.8272962050191,1084.549558284529,2.7475462753114623,18.747546275311464,1767.8272962050191,1799.8272962050191,1068.549558284529,1100.549558284529
        |6,P0001.zarr,27,0,10.725314672914088,449.96301449433514,1299.2491602470323,2.725314672914088,18.725314672914088,433.96301449433514,465.96301449433514,1283.2491602470323,1315.2491602470323
        |7,P0001.zarr,27,0,10.657296048712963,588.8341492975928,1779.0925104235198,2.657296048712963,18.657296048712965,572.8341492975928,604.8341492975928,1763.0925104235198,1795.0925104235198
        |8,P0001.zarr,27,0,9.115923958076873,695.0466327278888,580.433214468191,1.1159239580768734,17.115923958076873,679.0466327278888,711.0466327278888,564.433214468191,596.433214468191
        |9,P0001.zarr,27,0,11.441984962283017,993.628498457172,1721.5149988651165,3.4419849622830174,19.441984962283016,977.628498457172,1009.628498457172,1705.5149988651165,1737.5149988651165
        |""".stripMargin
    
    test("Any (position, time) repeat in drift file is an error.") {
        
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
        
        forAll (Gen.zip(genThreshold(arbitrary[NonnegativeReal]), arbitrary[ExtantOutputHandler])) { (threshold, outputHandler) =>
            withTempDirectory{ (tmpdir: os.Path) => 
                val spotsFile = tmpdir / "spots.csv"
                os.write(spotsFile, spotsLines)
                val driftFile = tmpdir / "drift.csv"
                os.write(driftFile, fineDriftLines)
                val caught = intercept[Exception]{ runLabelAndFilter(
                    spotsFile = spotsFile, 
                    driftFile = driftFile, 
                    probeGroups = List(), 
                    minSpotSeparation = threshold, 
                    filteredOutputFile = FilteredOutputFile.fromPath(tmpdir / "filtered.csv"),
                    unfilteredOutputFile = UnfilteredOutputFile.fromPath(tmpdir / "unfiltered.csv"),
                    extantOutputHandler = outputHandler,
                    )
                }
                val expectedRepeats = List(("P0001.zarr", 2) -> List(2, 4), ("P0002.zarr", 0) -> List(6, 7, 8))
                caught.getMessage shouldEqual s"${expectedRepeats.length} repeated (pos, time) pairs: ${expectedRepeats}"
            }
        }
    }

    test("Drift lines must be from fine drift, not just coarse drift.") {
        val coarseDriftLines = """,frame,position,z_px_coarse,y_px_coarse,x_px_coarse
            |0,0,P0001.zarr,-2.0,8.0,-24.0
            |1,1,P0001.zarr,-2.0,8.0,-20.0
            |2,1,P0001.zarr,0.0,6.0,-16.0
            |3,2,P0001.zarr,0.0,6.0,-12.0
            |4,0,P0002.zarr,0.0,4.0,-8.0
            |5,0,P0002.zarr,0.0,2.0,-6.0
            |6,1,P0002.zarr,0.0,0.0,-6.0
            |7,1,P0002.zarr,0.0,0.0,-4.0
            |8,1,P0002.zarr,0.0,0.0,-2.0
            |8,2,P0002.zarr,0.0,0.0,-4.0
            |""".stripMargin

        pending
    }

    test("The collection of ROIs parsed from filtered output is identical to that of the subset of input that has no proximal neighbors.") { pending }

    test("The collection of ROIs parsed from input is identical to the collection of ROIs parsed from unfiltered, labeled output.") { pending }

    test("Spot distance comparison operates in real space, NOT downsampled space. #143") { pending }

    test("Spot distance comparison responds to change of proximity comparison strategy #146.") { pending }

    // This tests for both the ability to specify nothing for the grouping, and for the correctness of the definition of the partitioning (trivial) when no grouping is specified.
    test("Spot distance comparison considers all ROIs as one big group if no grouping is provided. #147") {
        pending
    }
    
    test("In each pair of proximal spots, BOTH are filtered OUT. #148") { pending }
    
    test("Probe grouping must partition regional barcode frames: A probe grouping declaration that does not cover regional barcodes set is an error.") { pending }
    
    test("Probe grouping must partition regional barcode frames: A probe grouping declaration with an overlap (probe/frame repeated between groups) an error.") { pending }

    test("More proximal neighbors from different FOVs don't pair, while less proximal ones from the same FOV do pair. #150") { pending }

    test("ROI grouping by frame/probe/timepoint is specific to field-of-view, so that ROIs from different FOVs don't affect each other for filtering. #150") {
        
        /* Create all partitions of 5 as 2 and 3, mapping each partition to a position value for ROIs to be made. */
        val roiIndices = 0 until 5
        val partitions = roiIndices.combinations(3).map{
            group => roiIndices.toList.map(i => s"P000${if group.contains(i) then 1 else 2}.zarr")
        }
        
        /* Assert property for all partitions (on position) of 5 ROIs into a group of 2 and group of 3. */
        forAll (Table("positions", partitions.toList*)) { case positionAssigments =>
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
            forAll (genThreshold(arbitrary[NonnegativeReal]), minSuccessful(10)) {
                t => buildNeighboringRoisFinder(roisWithIndex, t)(List()) match {
                    case Left(errMsg) => fail(s"Expected test success but got failure/error message: $errMsg")
                    case Right(observed) => observed shouldEqual expected
                }
            }
        }
    }

    test("Zero threshold guarantees no neighbors.") {
        given arbitraryDoubleTemp: Arbitrary[Double] = genReasonableCoordinate.toArbitrary
        forAll (Gen.zip(genThreshold(NonnegativeReal(0)), arbitrary[List[Point3D]])) { 
            case (threshold, points) => 
                buildNeighborsLookupFlat(identity[Point3D])(points, threshold) shouldEqual Map()
        }
    }

    test("Total uniqueness of keys guarantees no neighbors.") {
        forAll (Gen.zip(genThreshold(NonnegativeReal(Double.PositiveInfinity)), arbitrary[List[Point3D]].map(_.zipWithIndex))) { 
            case (minDist, points) => buildNeighborsLookupKeyed(identity[Point3D])(points.map(_.swap), minDist) shouldEqual Map()
        }
    }

    test("All-singleton probe groupings guarantees no neighbors.") {
        def genRois: Gen[List[Roi]] = for {
            n <- Gen.choose(0, 10)
            regions <- Gen.pick(n, 0 until 100)
        } yield regions.map(r => canonicalRoi.copy(time = FrameIndex.unsafe(r))).toList
        
        forAll (genThreshold(NonnegativeReal(0)), genRois) { case (threshold, rois) => 
            val indexed = NonnegativeInt.indexed(rois)
            val grouping = rois.map(roi => ProbeGroup(NonEmptySet.one(roi.time)))
            buildNeighboringRoisFinder(indexed, threshold)(grouping) match {
                case Right(neigbors) => neigbors shouldEqual Map()
                case Left(errMsg) => fail(s"Expected success, but got error message: $errMsg")
            }
        }
    }

    test("Coincident ROIs in the same FOV are all mutually neighbors exactly when distance threshold is strictly positive.") { pending }

    test("Fewer than 2 ROIs means the neighbors mapping is always empty.") {
        pending
    }

    test("A ROI is never among its own neighbors.") {
        forAll (genThresholdAndRoisToFacilitateCollisions) { case (threshold, rois) => 
            buildNeighboringRoisFinder(NonnegativeInt.indexed(rois), threshold)(List()) match {
                case Left(errMsg) => fail(s"Expected success, but got error message: $errMsg")
                case Right(neighbors) => neighbors.toList.filter{ case (k, vs) => vs `contains` k } shouldEqual List()
            }
        }
    }

    test("Neighbor relation is bidirectional relation, so each of a ROI's neighbors has the ROI among its own neighbors.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        forAll (genThresholdAndRoisToFacilitateCollisions) { case (threshold, rois) => 
            buildNeighboringRoisFinder(NonnegativeInt.indexed(rois), threshold)(List()) match {
                case Left(errMsg) => fail(s"Expected success, but got error message: $errMsg")
                case Right(neighbors) => 
                    val (fails, _) = Alternative[List].separate(neighbors.toList.flatMap{ 
                        case curr@(k, vs) => vs.toList.map{ v => neighbors(v).contains(k).either(curr -> (v -> neighbors(v)), ()) }
                    })
                    fails shouldEqual List()
            }
        }
    }

    def genThresholdAndRoisToFacilitateCollisions: Gen[(DistanceThreshold, List[Roi])] = {
        val maxMinDistThreshold = NonnegativeReal(10) // Relatively small value to also use as upper bound on coordinates
        for {
            threshold <- genThreshold(genNonNegReal(maxMinDistThreshold))
            numRois <- Gen.choose(5, 10)
            centroids <- {
                given tmpArb: Arbitrary[Double] = Arbitrary(genNonNegReal(NonnegativeReal(5)))
                Gen.listOfN(numRois, arbitrary[Point3D])
            }
        } yield (threshold, centroids.map(canonicalRoi))
    }

    private def buildInterval[C <: Coordinate: [C] =>> NotGiven[C =:= Coordinate]](c: C, margin: NonnegativeReal)(lift: Double => C): BoundingBox.Interval[C] = 
        BoundingBox.Interval[C].apply.tupled((c.get - margin, c.get + margin).mapBoth(lift))

    private def canonicalRoi: Roi = canonicalRoi(Point3D(XCoordinate(1), YCoordinate(2), ZCoordinate(3)))

    private def canonicalRoi(point: Point3D): Roi = 
        point match { case Point3D(x, y, z) => 
            val xIntv = buildInterval(x, NonnegativeReal(2))(XCoordinate.apply)
            val yIntv = buildInterval(y, NonnegativeReal(2))(YCoordinate.apply)
            val zIntv = buildInterval(z, NonnegativeReal(1))(ZCoordinate.apply)
            val box = BoundingBox(xIntv, yIntv, zIntv)
            Roi(RoiIndex(NonnegativeInt(0)), "P0001.zarr", FrameIndex(NonnegativeInt(0)), Channel(NonnegativeInt(0)), point, box)
        }

    private def genThreshold: Gen[NonnegativeReal] => Gen[DistanceThreshold] = genThresholdType <*> _

    private def genThresholdType = Gen.oneOf(
        PiecewiseDistance.ConjunctiveThreshold.apply, 
        PiecewiseDistance.DisjunctiveThreshold.apply, 
        EuclideanDistance.Threshold.apply,
        )

end TestLabelAndFilterRois
