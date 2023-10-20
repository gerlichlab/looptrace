package at.ac.oeaw.imba.gerlich.looptrace

import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

import cats.Functor
import cats.syntax.flatMap.*
import cats.syntax.functor.*

import at.ac.oeaw.imba.gerlich.looptrace.space.*

trait LooptraceSuite:
    given arbitraryFunctor: Functor[Arbitrary] = new Functor[Arbitrary] {
        def map[A, B](fa: Arbitrary[A])(f: A => B): Arbitrary[B] = Arbitrary{
            arbitrary[A](fa).map(f)
        }
    }
    given roiIndexArbitrary(using idx: Arbitrary[Int]): Arbitrary[RoiIndex] = 
        Arbitrary{ arbitrary[Int].suchThat(_ >= 0).map(RoiIndex.apply compose NonnegativeInt.unsafe) }
    given xArbitrary(using num: Arbitrary[Double]): Arbitrary[XCoordinate] = num.fmap(XCoordinate.apply)
    given yArbitrary(using num: Arbitrary[Double]): Arbitrary[YCoordinate] = num.fmap(YCoordinate.apply)
    given zArbitrary(using num: Arbitrary[Double]): Arbitrary[ZCoordinate] = num.fmap(ZCoordinate.apply)
    given point3DArbitrary(using arbX: Arbitrary[Double]): Arbitrary[Point3D] = Arbitrary{
        Gen.zip(arbitrary[XCoordinate], arbitrary[YCoordinate], arbitrary[ZCoordinate]).map(Point3D.apply.tupled)
    }
    def genSelectedRoi[R <: SelectedRoi](build: (RoiIndex, Point3D) => R): Gen[R] = for {
        i <- arbitrary[RoiIndex]
        p <- arbitrary[Point3D]
    } yield build(i, p)
    given shiftingRoiArbitrary: Arbitrary[RoiForShifting] = Arbitrary{ genSelectedRoi(RoiForShifting.apply) }
    given accuracyRoiArbitrary: Arbitrary[RoiForAccuracy] = Arbitrary{ genSelectedRoi(RoiForAccuracy.apply) }
    given coordseqArbitrary: Arbitrary[CoordinateSequence] = 
        Arbitrary{ Gen.oneOf(CoordinateSequence.Forward, CoordinateSequence.Reverse) }
