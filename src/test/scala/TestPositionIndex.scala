package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Random
import cats.syntax.eq.*
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should

/** Tests for positive integer refinement type */
class TestPositionIndex extends AnyFunSuite, GenericSuite, ScalacheckSuite, should.Matchers:
    
    test("Position indices are equivalent on their wrapped values.") {
        def genEqv = arbitrary[NonnegativeInt].map{ n => (PositionIndex(n), PositionIndex(n), true) }
        
        def genNonEqv = for {
            n1 <- arbitrary[NonnegativeInt]
            n2 <- arbitrary[NonnegativeInt].suchThat(_ =!= n1)
        } yield (PositionIndex(n1), PositionIndex(n2), false)
        
        forAll (Gen.oneOf(genEqv, genNonEqv)) { case (p1, p2, exp) => p1 === p2 shouldBe exp }
    }

    test("Set respects position index equivalence.") {
        def genValuesAndNumUnique = for {
            numUniq <- Gen.choose(1, 5)
            uniqueValues <- Gen.listOfN(numUniq, Gen.choose(0, 100)).suchThat(zs => zs.length === zs.toSet.size)
            repeatCounts <- Gen.listOfN(uniqueValues.length, Gen.choose(2, 5))
            values = uniqueValues.zip(repeatCounts).flatMap{ (z, rep) => List.fill(rep)(z) }
        } yield (Random.shuffle(values.map(unsafeLiftInt)), values.toSet.map(unsafeLiftInt))
        
        forAll (genValuesAndNumUnique) { case (indices, expected) => indices.toSet shouldEqual expected }
    }

    def unsafeLiftInt = NonnegativeInt.unsafe.andThen(PositionIndex.apply)

end TestPositionIndex