package at.ac.oeaw.imba.gerlich.looptrace

import cats.Applicative
import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

/** Fairly generalised helpers for Scalcheck */
trait ScalacheckGenericExtras:

    /** Define mapping operation by building new arbitrary after mapping over the instance's generator. */
    given applicativeForArbitrary(using ev: Applicative[Gen]): Applicative[Arbitrary] with
        override def pure[A](a: A): Arbitrary[A] = ev.pure(a).toArbitrary
        override def ap[A, B](ff: Arbitrary[A => B])(fa: Arbitrary[A]): Arbitrary[B] = 
            ev.ap(ff.arbitrary)(fa.arbitrary).toArbitrary

    /** Use Gen.flatMap to define {@code Applicative.ap}, and {@code Gen.const} to define {@code Applicative.pure}. */
    given applicativeForGen: Applicative[Gen] with
        override def pure[A](a: A) = Gen.const(a)
        override def ap[A, B](ff: Gen[A => B])(fa: Gen[A]): Gen[B] = for {
            f <- ff
            a <- fa
        } yield f(a)

    /** Add nicer syntax to arbitrary instances. */
    extension [A](arb: Arbitrary[A])
        infix def suchThat(p: A => Boolean): Arbitrary[A] = (arb.arbitrary `suchThat` p).toArbitrary

    /** Add nicer syntax to generators. */
    extension [A](g: Gen[A])
        def toArbitrary: Arbitrary[A] = Arbitrary(g)
        infix def zipWith[B](b: B): Gen[(A, B)] = g.map(_ -> b)

end ScalacheckGenericExtras
