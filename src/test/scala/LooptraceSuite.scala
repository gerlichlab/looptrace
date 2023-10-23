package at.ac.oeaw.imba.gerlich.looptrace

import cats.Functor
import cats.syntax.functor.*

import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

import at.ac.oeaw.imba.gerlich.looptrace.space.*

/** Base trait for tests in looptrace */
trait LooptraceSuite:
    
    /************************/
    /* Givens ("implicits") */
    /************************/
    given arbitraryFunctor: Functor[Arbitrary] with
        def map[A, B](fa: Arbitrary[A])(f: A => B): Arbitrary[B] = 
            Arbitrary{ arbitrary[A](fa).map(f) }

    /** Zip together 2 arbitrary instances */
    given arbZip2[A, B](using Arbitrary[A], Arbitrary[B]): Arbitrary[(A, B)] = Arbitrary{
        for {
            a <- arbitrary[A]
            b <- arbitrary[B]
        } yield (a, b)
    }

    /** Zip together 3 arbitrary instances */
    given arbZip3[A, B, C](using Arbitrary[A], Arbitrary[B], Arbitrary[C]): Arbitrary[(A, B, C)] = Arbitrary{
        for {
            a <- arbitrary[A]
            b <- arbitrary[B]
            c <- arbitrary[C]
        } yield (a, b, c)
    }

    given roiIndexArbitrary(using idx: Arbitrary[Int]): Arbitrary[RoiIndex] = 
        Arbitrary{ Gen.choose(0, Int.MaxValue).map(RoiIndex.apply compose NonnegativeInt.unsafe) }
    
    given xArbitrary(using num: Arbitrary[Double]): Arbitrary[XCoordinate] = num.fmap(XCoordinate.apply)
    
    given yArbitrary(using num: Arbitrary[Double]): Arbitrary[YCoordinate] = num.fmap(YCoordinate.apply)
    
    given zArbitrary(using num: Arbitrary[Double]): Arbitrary[ZCoordinate] = num.fmap(ZCoordinate.apply)
    
    given point3DArbitrary(using arbX: Arbitrary[Double]): Arbitrary[Point3D] = Arbitrary{
        Gen.zip(arbitrary[XCoordinate], arbitrary[YCoordinate], arbitrary[ZCoordinate]).map(Point3D.apply.tupled)
    }
    
    given coordseqArbitrary: Arbitrary[CoordinateSequence] = 
        Arbitrary{ Gen.oneOf(CoordinateSequence.Forward, CoordinateSequence.Reverse) }
    
    given nonnegativeIntArbitray: Arbitrary[NonnegativeInt] = Arbitrary { genNonnegativeInt }
    
    given positiveIntArbitrary: Arbitrary[PositiveInt] = Arbitrary{ genPositiveInt }

    /********************/
    /* Other defintions */
    /********************/
    def genNonnegativeInt: Gen[NonnegativeInt] = Gen.choose(0, Int.MaxValue).map(NonnegativeInt.unsafe)
    
    def genPositiveInt: Gen[PositiveInt] = Gen.posNum[Int].map(PositiveInt.unsafe)
    
    def touchFile(fp: os.Path): Unit = os.write(fp, "")

    /** Execute some test code that uses a {@code os.Path} folder. */
    def withTempDirectory(testCode: os.Path => Any): Any = {
        val tempRoot = os.temp.dir()
        try { testCode(tempRoot) } finally { os.remove.all(tempRoot) }
    }

    /** Execute some test code that uses a {@code os.Path} file. */
    def withTempFile(initData: os.Source = null)(testCode: os.Path => Any): Any = {
        val tempfile = os.temp(contents = initData)
        try { testCode(tempfile) } finally { os.remove(tempfile) }
    }
