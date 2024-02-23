package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import cats.syntax.all.*
import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

import at.ac.oeaw.imba.gerlich.looptrace.space.*

/** Base trait for tests in looptrace */
trait LooptraceSuite extends GenericSuite, ScalacheckGenericExtras:

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

    given arbitraryForChannel(using arbInt: Arbitrary[NonnegativeInt]): Arbitrary[Channel] = arbInt.map(Channel.apply)

    given arbitraryForLocusId(using arbTime: Arbitrary[Timepoint]): Arbitrary[LocusId] = arbTime.map(LocusId.apply)

    given arbitraryForPositionIndex(using idx: Arbitrary[NonnegativeInt]): Arbitrary[PositionIndex] = idx.map(PositionIndex.apply)

    given arbitraryForPositionName: Arbitrary[PositionName] = 
        Arbitrary{ Gen.alphaNumStr.suchThat(_.nonEmpty).map(PositionName.apply) }

    given arbitraryForProbeName(using arbName: Arbitrary[String]): Arbitrary[ProbeName] = arbName.suchThat(_.nonEmpty).map(ProbeName.apply)

    given arbitraryForRegionId(using arbTime: Arbitrary[Timepoint]): Arbitrary[RegionId] = arbTime.map(RegionId.apply)

    given arbitraryForRoiIndex(using idx: Arbitrary[Int]): Arbitrary[RoiIndex] = 
        Arbitrary{ Gen.choose(0, Int.MaxValue).map(RoiIndex.unsafe) }
    
    given arbitraryForTimepoint(using idx: Arbitrary[NonnegativeInt]): Arbitrary[Timepoint] = idx.map(Timepoint.apply)

    given arbitraryForXCoordinate(using num: Arbitrary[Double]): Arbitrary[XCoordinate] = num.fmap(XCoordinate.apply)
    
    given arbitraryForYCoordinate(using num: Arbitrary[Double]): Arbitrary[YCoordinate] = num.fmap(YCoordinate.apply)
    
    given arbitraryForZCoordinate(using num: Arbitrary[Double]): Arbitrary[ZCoordinate] = num.fmap(ZCoordinate.apply)
    
    given arbitraryForPoint3D(using arbX: Arbitrary[Double]): Arbitrary[Point3D] = Arbitrary{
        Gen.zip(arbitrary[XCoordinate], arbitrary[YCoordinate], arbitrary[ZCoordinate]).map(Point3D.apply.tupled)
    }
    
    given arbitraryForRegionGroupingSemantic: Arbitrary[ImagingRoundsConfiguration.RegionGrouping.Semantic] = Gen.oneOf(
        ImagingRoundsConfiguration.RegionGrouping.Semantic.Permissive, 
        ImagingRoundsConfiguration.RegionGrouping.Semantic.Prohibitive, 
        ).toArbitrary

    given arbitraryForBlankImagingRound(using arbName: Arbitrary[String], arbTime: Arbitrary[Timepoint]): Arbitrary[BlankImagingRound] = 
        (arbName, arbTime).mapN(BlankImagingRound.apply)

    given arbitraryForRegionalImagingRound(using 
        arbName: Arbitrary[String], 
        arbTime: Arbitrary[Timepoint], 
        arbProbe: Arbitrary[ProbeName]
        ): Arbitrary[RegionalImagingRound] = (arbName, arbTime, arbProbe).mapN(RegionalImagingRound.apply)

    given arbitraryForLocusImagingRound(using 
        arbName: Arbitrary[String], 
        arbTime: Arbitrary[Timepoint], 
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
        arbCh: Arbitrary[Channel], 
        arbPt: Arbitrary[Point3D],
        arbMargin: Arbitrary[BoundingBox.Margin],
        ): Arbitrary[RegionalBarcodeSpotRoi] = {
        def genRoi: Gen[RegionalBarcodeSpotRoi] = for {
            idx <- arbitrary[RoiIndex]
            pos <- arbitrary[PositionName]
            reg <- arbitrary[RegionId]
            ch <- arbitrary[Channel]
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
    protected def buildRectangularBox(pt: Point3D)(xMargin: BoundingBox.Margin, yMargin: BoundingBox.Margin, zMargin: BoundingBox.Margin): BoundingBox = {
        def buildInterval[C <: Coordinate : [C] =>> NotGiven[C =:= Coordinate]](lift: Double => C)(center: Double, margin: BoundingBox.Margin): BoundingBox.Interval[C] = 
            BoundingBox.Interval.apply[C].tupled((center - margin.get, center + margin.get).mapBoth(lift))
        val ix = buildInterval(XCoordinate.apply)(pt.x.get, xMargin)
        val iy = buildInterval(YCoordinate.apply)(pt.y.get, yMargin)
        val iz = buildInterval(ZCoordinate.apply)(pt.z.get, zMargin)
        BoundingBox(ix, iy, iz)
    }

end LooptraceSuite