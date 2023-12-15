package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import cats.*
import cats.data.{ NonEmptySet }
import cats.syntax.all.*
import cats.data.NonEmptySet
import mouse.boolean.*
import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*

import at.ac.oeaw.imba.gerlich.looptrace.space.{ 
    Coordinate, 
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
    BoundingBox, 
    LineNumber, 
    ProbeGroup,
    Roi, 
    RoiIdxPair
}
import at.ac.oeaw.imba.gerlich.looptrace.space.DistanceThreshold
import cats.data.NonEmptySeq

/** Tests for the filtration of the individual supports (single FISH probes) of chromatin fiber traces */
class TestLabelAndFilterRois extends AnyFunSuite, DistanceSuite, LooptraceSuite, ScalacheckSuite, should.Matchers:

    test("Spot distance comparison uses drift correction.") { pending }
    
    test("The collection of ROIs parsed from filtered output is identical to that of the subset of input that has no proximal neighbors.") { pending }

    test("The collection of ROIs parsed from input is identical to the collection of ROIs parsed from unfiltered, labeled output.") { pending }

    test("Spot distance comparison responds to change of proximity comparison strategy #146.") { pending }

    // This tests for both the ability to specify nothing for the grouping, and for the correctness of the definition of the partitioning (trivial) when no grouping is specified.
    test("Spot distance comparison considers all ROIs as one big group if no grouping is provided. #147") {
        pending
    }
    
    test("In each pair of proximal spots, BOTH are filtered OUT. #148") { pending }
    
    test("Probe grouping must partition regional barcode frames: A probe grouping declaration that does not cover regional barcodes set is an error.") { pending }
    
    test("Probe grouping must partition regional barcode frames: A probe grouping declaration with an overlap (probe/frame repeated between groups) an error.") { pending }

    test("ROI grouping by frame/probe/timepoint is specific to field-of-view, so that ROIs from different FOVs don't affect each other for filtering.") {
        
        /* Create all partitions of 5 as 2 and 3, mapping each partition to a position value for ROIs to be made. */
        val roiIndices = 0 until 5
        val partitions = roiIndices.combinations(3).map{
            group => roiIndices.toList.map(i => s"P000${if group.contains(i) then 1 else 2}.zarr")
        }
        
        /* Assert property for all partitions (on position) of 5 ROIs into a group of 2 and group of 3 */
        forAll (Table("positions", partitions.toList*)) { case positionAssigments =>
            /* Build ROIs with position assigned as randomised, and with index within list. */
            val roisWithIndex = NonnegativeInt.indexed(positionAssigments).map(_.leftMap{ pos => canonicalRoi.copy(position = pos) })
            /* Create the expected neighboring ROIs grouping, based on fact that all ROIs within each position group are coincident. */
            given orderForPair: Order[RoiIdxPair] = Order.by(_._2)
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

    private def genThresholdType = Gen.oneOf(PiecewiseDistance.DisjunctiveThreshold.apply, EuclideanDistance.Threshold.apply)

end TestLabelAndFilterRois
