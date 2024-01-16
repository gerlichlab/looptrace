package at.ac.oeaw.imba.gerlich.looptrace

import cats.Applicative
import cats.data.NonEmptyList
import cats.syntax.all.*
import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

/** Fairly generalised helpers for Scalcheck */
trait ScalacheckGenericExtras:

    /** Define {@code Applicative[Arbitrary]} i.t.o. {@code Applicative[Gen]}. */
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

    /** Generate a nonempty list and then convert it to an arbitrary instance. */
    given arbitraryForNonEmptyList[A : Arbitrary]: Arbitrary[NonEmptyList[A]] = 
        Gen.nonEmptyListOf(arbitrary[A]).map(_.toNel.get).toArbitrary

    /** Add nicer syntax to arbitrary instances. */
    extension [A](arb: Arbitrary[A])
        infix def suchThat(p: A => Boolean): Arbitrary[A] = (arb.arbitrary `suchThat` p).toArbitrary

    /** Add nicer syntax to generators. */
    extension [A](g: Gen[A])
        def toArbitrary: Arbitrary[A] = Arbitrary(g)
        infix def zipWith[B](b: B): Gen[(A, B)] = g.map(_ -> b)

end ScalacheckGenericExtras
