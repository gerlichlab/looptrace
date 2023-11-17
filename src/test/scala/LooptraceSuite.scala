package at.ac.oeaw.imba.gerlich.looptrace

import cats.syntax.functor.*

import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

import at.ac.oeaw.imba.gerlich.looptrace.space.*

/** Base trait for tests in looptrace */
trait LooptraceSuite extends GenericSuite, ScalacheckGenericExtras:

    /************************/
    /* Givens ("implicits") */
    /************************/

    /** Zip together 3 arbitrary instances */
    given arbZip3[A, B, C](using Arbitrary[A], Arbitrary[B], Arbitrary[C]): Arbitrary[(A, B, C)] = Arbitrary{
        for {
            a <- arbitrary[A]
            b <- arbitrary[B]
            c <- arbitrary[C]
        } yield (a, b, c)
    }

    given frameIndexArbitrary(using idx: Arbitrary[NonnegativeInt]): Arbitrary[FrameIndex] = idx.map(FrameIndex.apply)

    given positionIndexArbitrary(using idx: Arbitrary[NonnegativeInt]): Arbitrary[PositionIndex] = idx.map(PositionIndex.apply)

    given roiIndexArbitrary(using idx: Arbitrary[Int]): Arbitrary[RoiIndex] = 
        Arbitrary{ Gen.choose(0, Int.MaxValue).map(RoiIndex.unsafe) }
    
    given coordseqArbitrary: Arbitrary[CoordinateSequence] = 
        Arbitrary{ Gen.oneOf(CoordinateSequence.Forward, CoordinateSequence.Reverse) }

    given delimiterArbitrary: Arbitrary[Delimiter] = Arbitrary{ Gen.oneOf(Delimiter.CommaSeparator, Delimiter.TabSeparator) }

    given xArbitrary(using num: Arbitrary[Double]): Arbitrary[XCoordinate] = num.fmap(XCoordinate.apply)
    
    given yArbitrary(using num: Arbitrary[Double]): Arbitrary[YCoordinate] = num.fmap(YCoordinate.apply)
    
    given zArbitrary(using num: Arbitrary[Double]): Arbitrary[ZCoordinate] = num.fmap(ZCoordinate.apply)
    
    given point3DArbitrary(using arbX: Arbitrary[Double]): Arbitrary[Point3D] = Arbitrary{
        Gen.zip(arbitrary[XCoordinate], arbitrary[YCoordinate], arbitrary[ZCoordinate]).map(Point3D.apply.tupled)
    }
