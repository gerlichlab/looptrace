package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import cats.syntax.all.*
import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

import at.ac.oeaw.imba.gerlich.gerlib.geometry.BoundingBox
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.testing.{ GeometricInstances, ImagingInstances, CatsScalacheckInstances }
import at.ac.oeaw.imba.gerlich.gerlib.testing.syntax.SyntaxForScalacheck

import at.ac.oeaw.imba.gerlich.looptrace.space.{
    BoundingBox as BB,
    Coordinate, 
    CoordinateSequence, 
    EuclideanDistance,
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

    given arbitraryForExtantOutputHandler: Arbitrary[ExtantOutputHandler] = 
        Arbitrary{ Gen.oneOf(ExtantOutputHandler.values.toIndexedSeq) }

    given arbitraryForLocusId(using arbTime: Arbitrary[ImagingTimepoint]): Arbitrary[LocusId] = arbTime.map(LocusId.apply)

    given arbitraryForPositionIndex(using idx: Arbitrary[NonnegativeInt]): Arbitrary[PositionIndex] = idx.map(PositionIndex.apply)

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

    given arbitraryForRegionalBarcodeSpotRoi(using
        arbRoiIdx: Arbitrary[RoiIndex], 
        arbPosName: Arbitrary[PositionName], 
        arbRegion: Arbitrary[RegionId], 
        arbCh: Arbitrary[ImagingChannel], 
        arbPt: Arbitrary[Point3D],
        arbMargin: Arbitrary[BoundingBox.Margin],
        ): Arbitrary[RegionalBarcodeSpotRoi] = {
        def genRoi: Gen[RegionalBarcodeSpotRoi] = for {
            idx <- arbitrary[RoiIndex]
            pos <- arbitrary[PositionName]
            reg <- arbitrary[RegionId]
            ch <- arbitrary[ImagingChannel]
            pt <- arbitrary[Point3D]
            box <- arbitrary[(BoundingBox.Margin, BoundingBox.Margin, BoundingBox.Margin)].map(buildRectangularBox(pt).tupled)
        } yield RegionalBarcodeSpotRoi(index = idx, position = pos, region = reg, channel = ch, centroid = pt, boundingBox = box)
        Arbitrary(genRoi)
    }

    /************************
     * Other definitions
     ***********************/
    protected def genNonNegInt(limit: NonnegativeInt): Gen[NonnegativeInt] = Gen.choose(0, limit).map(NonnegativeInt.unsafe)
    protected def genNonNegReal(limit: NonnegativeReal): Gen[NonnegativeReal] = Gen.choose(0.0, limit).map(NonnegativeReal.unsafe)
    protected def genPosReal(limit: PositiveReal): Gen[PositiveReal] = Gen.choose(0.0, limit).suchThat(_ > 0).map(PositiveReal.unsafe)
    protected def buildRectangularBox(pt: Point3D)(xMargin: BoundingBox.Margin, yMargin: BoundingBox.Margin, zMargin: BoundingBox.Margin): BB = {
        def buildInterval[C <: Coordinate : [C] =>> NotGiven[C =:= Coordinate]](lift: Double => C)(center: Double, margin: BoundingBox.Margin): BoundingBox.Interval[Double, C] = 
            BoundingBox.Interval.apply[Double, C].tupled((center - margin.get, center + margin.get).mapBoth(lift))
        val ix = buildInterval(XCoordinate.apply)(pt.x.get, xMargin)
        val iy = buildInterval(YCoordinate.apply)(pt.y.get, yMargin)
        val iz = buildInterval(ZCoordinate.apply)(pt.z.get, zMargin)
        BoundingBox(ix, iy, iz)
    }

end LooptraceSuite