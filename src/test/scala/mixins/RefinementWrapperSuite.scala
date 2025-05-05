package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Random
import cats.Eq
import cats.syntax.eq.*
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.{Arbitrary, Gen}

/** Helpers for tests for a simple wrapper type (often extending {@code AnyVal})
  */
trait RefinementWrapperSuite extends GenericSuite:

  def ironNonnegativityFailureMessage = "!(Should be strictly negative)"

  /** Generate either two expected-equivalent or expected-nonequivalent values,
    * and expected equivalence test result.
    */
  def genEquivalenceInputAndExpectation[A: {Arbitrary, Eq}, B](
      lift: A => B
  ): Gen[(B, B, Boolean)] = {
    def genEqv = arbitrary[A].map { a => (lift(a), lift(a), true) }
    def genNonEqv = Gen
      .zip(arbitrary[A], arbitrary[A])
      .suchThat(_ =!= _)
      .map((a1, a2) => (lift(a1), lift(a2), false))
    Gen.oneOf(genEqv, genNonEqv)
  }

  /** Generate a multiset (as {@code List}) and the corresponding expectation
    * after calling {@code .toSet} on the multiset.
    */
  def genValuesAndNumUnique[A, B](
      genA: Gen[A]
  )(lift: A => B): Gen[(List[B], Set[B])] = for
    numUniq <- Gen.choose(1, 5)
    uniqueValues <- Gen
      .listOfN(numUniq, genA)
      .suchThat(zs => zs.length === zs.toSet.size)
    repeatCounts <- Gen.listOfN(uniqueValues.length, Gen.choose(2, 5))
    values = uniqueValues.zip(repeatCounts).flatMap { (z, rep) =>
      List.fill(rep)(z)
    }
  yield (Random.shuffle(values.map(lift)), values.toSet.map(lift))

end RefinementWrapperSuite
