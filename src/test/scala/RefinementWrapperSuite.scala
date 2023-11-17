package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Random
import cats.Eq
import cats.syntax.eq.*
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.{ Arbitrary, Gen }

trait RefinementWrapperSuite extends GenericSuite:

    def genEquivalenceInputAndExpectation[A : Arbitrary : Eq, B](lift: A => B): Gen[(B, B, Boolean)] = {
        def genEqv = arbitrary[A].map{ a => (lift(a), lift(a), true) }
        def genNonEqv = Gen.zip(arbitrary[A], arbitrary[A]).suchThat(_ =!= _).map(t => (lift(t._1), lift(t._2), false))
        Gen.oneOf(genEqv, genNonEqv)
    }

    def genValuesAndNumUnique[A](lift: Int => A): Gen[(List[A], Set[A])] = for {
        numUniq <- Gen.choose(1, 5)
        uniqueValues <- Gen.listOfN(numUniq, Gen.choose(0, 100)).suchThat(zs => zs.length === zs.toSet.size)
        repeatCounts <- Gen.listOfN(uniqueValues.length, Gen.choose(2, 5))
        values = uniqueValues.zip(repeatCounts).flatMap{ (z, rep) => List.fill(rep)(z) }
    } yield (Random.shuffle(values.map(lift)), values.toSet.map(lift))

end RefinementWrapperSuite