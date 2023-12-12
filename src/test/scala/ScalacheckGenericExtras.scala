package at.ac.oeaw.imba.gerlich.looptrace

import cats.{ Applicative, Functor }
import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

/** Fairly generalised helpers for Scalcheck */
trait ScalacheckGenericExtras:

    /** Define mapping operation by building new arbitrary after mapping over the instance's generator. */
    given functorForArbitrary: Functor[Arbitrary] with
        override def map[A, B](arb: Arbitrary[A])(f: A => B): Arbitrary[B] = 
            Arbitrary{ arbitrary[A](arb).map(f) }

    /** Use Gen.flatMap to define {@code Applicative.ap}, and {@code Gen.const} to define {@code Applicative.pure}. */
    given applicativeForGen: Applicative[Gen] with
        override def pure[A](a: A) = Gen.const(a)
        override def ap[A, B](ff: Gen[A => B])(fa: Gen[A]): Gen[B] = for {
            f <- ff
            a <- fa
        } yield f(a)

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

    /** Add nicer syntax to arbitrary instances. */
    extension [A](arb: Arbitrary[A])
        def gen: Gen[A] = arb.arbitrary
        infix def suchThat(p: A => Boolean): Arbitrary[A] = Arbitrary{ gen `suchThat` p }

    /** Add nicer syntax to generators. */
    extension [A](g: Gen[A])
        def toArbitrary: Arbitrary[A] = Arbitrary(g)
        infix def zipWith[B](b: B): Gen[(A, B)] = g.map(_ -> b)

end ScalacheckGenericExtras
