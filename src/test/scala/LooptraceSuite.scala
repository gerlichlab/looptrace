package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import cats.syntax.functor.*
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

    given arbitraryForFrameIndex(using idx: Arbitrary[NonnegativeInt]): Arbitrary[FrameIndex] = idx.map(FrameIndex.apply)

    given arbitraryForPositionIndex(using idx: Arbitrary[NonnegativeInt]): Arbitrary[PositionIndex] = idx.map(PositionIndex.apply)

    given arbitraryForRoiIndex(using idx: Arbitrary[Int]): Arbitrary[RoiIndex] = 
        Arbitrary{ Gen.choose(0, Int.MaxValue).map(RoiIndex.unsafe) }
    
    given arbitraryForXCoordinate(using num: Arbitrary[Double]): Arbitrary[XCoordinate] = num.fmap(XCoordinate.apply)
    
    given arbitraryForYCoordinate(using num: Arbitrary[Double]): Arbitrary[YCoordinate] = num.fmap(YCoordinate.apply)
    
    given arbitraryForZCoordinate(using num: Arbitrary[Double]): Arbitrary[ZCoordinate] = num.fmap(ZCoordinate.apply)
    
    given point3DArbitrary(using arbX: Arbitrary[Double]): Arbitrary[Point3D] = Arbitrary{
        Gen.zip(arbitrary[XCoordinate], arbitrary[YCoordinate], arbitrary[ZCoordinate]).map(Point3D.apply.tupled)
    }

    given arbitraryForRegionalBarcodeSpotRoi(
        using arbRoiIdx: Arbitrary[RoiIndex], 
        arbFrameIdx: Arbitrary[FrameIndex], 
        arbCh: Arbitrary[Channel], 
        arbPt: Arbitrary[Point3D], 
        arbMargin: Arbitrary[BoundingBox.Margin],
        ): Arbitrary[RegionalBarcodeSpotRoi] = {
        def buildBox(pt: Point3D)(xMargin: BoundingBox.Margin, yMargin: BoundingBox.Margin, zMargin: BoundingBox.Margin): BoundingBox = {
            def buildInterval[C <: Coordinate : [C] =>> NotGiven[C =:= Coordinate]](lift: Double => C)(center: Double, margin: Double): BoundingBox.Interval[C] = 
                BoundingBox.Interval.apply[C].tupled((center - margin, center + margin).mapBoth(lift))
            val ix = buildInterval(XCoordinate.apply)(pt.x.get, xMargin.get)
            val iy = buildInterval(YCoordinate.apply)(pt.y.get, yMargin.get)
            val iz = buildInterval(ZCoordinate.apply)(pt.z.get, zMargin.get)
            BoundingBox(ix, iy, iz)
        }
        def genRoi: Gen[RegionalBarcodeSpotRoi] = for {
            idx <- arbitrary[RoiIndex]
            pos <- Gen.alphaNumStr
            t <- arbitrary[FrameIndex]
            ch <- arbitrary[Channel]
            pt <- arbitrary[Point3D]
            box <- arbitrary[(BoundingBox.Margin, BoundingBox.Margin, BoundingBox.Margin)].map(buildBox(pt))
        } yield RegionalBarcodeSpotRoi(index = idx, position = pos, time = t, channel = ch, centroid = pt, boundingBox = box)
        Arbitrary(genRoi)
    }

    /************************
     * Other definitions
     ***********************/
    def genNonNegInt(limit: NonnegativeInt): Gen[NonnegativeInt] = Gen.choose(0, limit).map(NonnegativeInt.unsafe)
    def genNonNegReal(limit: NonnegativeReal): Gen[NonnegativeReal] = Gen.choose(0.0, limit).map(NonnegativeReal.unsafe)

end LooptraceSuite