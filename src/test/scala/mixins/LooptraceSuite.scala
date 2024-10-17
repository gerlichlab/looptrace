package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import cats.*
import cats.syntax.all.*
import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.geometry.{BoundingBox, Centroid, EuclideanDistance}
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.roi.DetectedSpot
import at.ac.oeaw.imba.gerlich.gerlib.testing.instances.{ GeometricInstances, ImagingInstances, CatsScalacheckInstances }
import at.ac.oeaw.imba.gerlich.gerlib.testing.syntax.SyntaxForScalacheck

import at.ac.oeaw.imba.gerlich.looptrace.roi.{ 
    DetectedSpotRoi, 
    MergedRoiRecord, 
}
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.{
    IndexedDetectedSpot, 
    PostMergeRoi,
}
import at.ac.oeaw.imba.gerlich.looptrace.space.{
    BoundingBox as BB,
    Coordinate, 
    CoordinateSequence,
    Point3D,
    XCoordinate, 
    YCoordinate, 
    ZCoordinate,
}
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Base trait for tests in looptrace */
trait LooptraceSuite extends GenericSuite, GeometricInstances, ImagingInstances, CatsScalacheckInstances, SyntaxForScalacheck:

    /************************/
    /* Givens ("implicits") */
    /************************/
    given arbitraryForCoordinateSequence: Arbitrary[CoordinateSequence] = 
        Arbitrary{ Gen.oneOf(CoordinateSequence.Forward, CoordinateSequence.Reverse) }

    given arbitraryForDelimiter: Arbitrary[Delimiter] = Arbitrary{ Gen.oneOf(Delimiter.CommaSeparator, Delimiter.TabSeparator) }

    given arbitraryForEuclideanThreshold(using arbT: Arbitrary[NonnegativeReal]): Arbitrary[EuclideanDistance.Threshold] = 
        arbT.map(EuclideanDistance.Threshold.apply)

    given arbitraryForLocusId(using arbTime: Arbitrary[ImagingTimepoint]): Arbitrary[LocusId] = arbTime.map(LocusId.apply)

    given arbitraryForProbeName(using arbName: Arbitrary[String]): Arbitrary[ProbeName] = arbName.suchThat(_.nonEmpty).map(ProbeName.apply)

    given arbitraryForRegionId(using arbTime: Arbitrary[ImagingTimepoint]): Arbitrary[RegionId] = arbTime.map(RegionId.apply)

    given arbitraryForRoiIndex(using idx: Arbitrary[Int]): Arbitrary[RoiIndex] = 
        Arbitrary{ Gen.choose(0, Int.MaxValue).map(RoiIndex.unsafe) }
    
    given arbitraryForBlankImagingRound(using arbName: Arbitrary[String], arbTime: Arbitrary[ImagingTimepoint]): Arbitrary[BlankImagingRound] = 
        (arbName, arbTime).mapN(BlankImagingRound.apply)

    given arbitraryForRegionalImagingRound(using 
        arbName: Arbitrary[String], 
        arbTime: Arbitrary[ImagingTimepoint], 
        arbProbe: Arbitrary[ProbeName]
        ): Arbitrary[RegionalImagingRound] = (arbName, arbTime, arbProbe).mapN(RegionalImagingRound.apply)

    given arbitraryForLocusImagingRound(using 
        arbName: Arbitrary[String], 
        arbTime: Arbitrary[ImagingTimepoint], 
        arbProbe: Arbitrary[ProbeName], 
        arbRepeat: Arbitrary[PositiveInt]
        ): Arbitrary[LocusImagingRound] = {
            val arbRepOpt = Gen.option(arbitrary(arbRepeat)).toArbitrary
            (arbName, arbTime, arbProbe, arbRepOpt).mapN(LocusImagingRound.apply)
        }

    given arbitraryForDetectedSpotRoi(using 
        arbSpot: Arbitrary[DetectedSpot[Double]], 
        arbBox: Arbitrary[BoundingBox[Double]],
    ): Arbitrary[DetectedSpotRoi] = (arbSpot, arbBox).mapN(DetectedSpotRoi.apply)

    given arbitraryForIndexedDetectedSpot(using 
        arbIndex: Arbitrary[RoiIndex],
        arbContext: Arbitrary[ImagingContext], 
        arbCentroid: Arbitrary[Centroid[Double]], 
        arbBox: Arbitrary[BB], 
    ): Arbitrary[IndexedDetectedSpot] = 
        (arbIndex, arbContext, arbCentroid, arbBox).mapN(IndexedDetectedSpot.apply)

    // TODO: use this to reconstruct proximity-based filtration tests.
    given arbitraryForPostMergeRoi(using 
        Arbitrary[IndexedDetectedSpot], 
        Arbitrary[MergedRoiRecord],
    ): Arbitrary[PostMergeRoi] = 
        Arbitrary.oneOf[IndexedDetectedSpot, MergedRoiRecord]
    
    /************************
     * Other definitions
     ***********************/
    protected def genNonNegReal(limit: NonnegativeReal): Gen[NonnegativeReal] = Gen.choose(0.0, limit).map(NonnegativeReal.unsafe)
    protected def genPosReal(limit: PositiveReal): Gen[PositiveReal] = Gen.choose(0.0, limit).suchThat(_ > 0).map(PositiveReal.unsafe)
end LooptraceSuite