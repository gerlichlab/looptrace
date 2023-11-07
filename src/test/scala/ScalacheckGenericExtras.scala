package at.ac.oeaw.imba.gerlich.looptrace

import cats.Functor
import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

/** Fairly generalised helpers for Scalcheck */
trait ScalacheckGenericExtras {

    /** Define mapping operation by building new arbitrary after mapping over the instance's generator. */
    given arbitraryFunctor: Functor[Arbitrary] with
        def map[A, B](arb: Arbitrary[A])(f: A => B): Arbitrary[B] = 
            Arbitrary{ arbitrary[A](arb).map(f) }

    /** Zip together 2 arbitrary instances */
    given arbZip2[A, B](using Arbitrary[A], Arbitrary[B]): Arbitrary[(A, B)] = Arbitrary{
        for {
            a <- arbitrary[A]
            b <- arbitrary[B]
        } yield (a, b)
    }

    /** Add nicer syntax to arbitrary instances. */
    implicit class ArbitraryOps[A](arb: Arbitrary[A]):
        def gen: Gen[A] = arb.arbitrary
        infix def suchThat(p: A => Boolean): Arbitrary[A] = Arbitrary{ gen `suchThat` p }

    /** Add nicer syntax to generators. */
    implicit class GeneratorOps[A](g: Gen[A]):
        infix def zipWith[B](b: B): Gen[(A, B)] = g.map(_ -> b)
}
