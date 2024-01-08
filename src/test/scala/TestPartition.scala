package at.ac.oeaw.imba.gerlich.looptrace

import cats.syntax.all.*
import org.scalacheck.{ Arbitrary, Gen, Shrink }
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.looptrace.collections.*

/**
 * Tests (over relatively small inputs) of generic cartesian product.
 * 
 * @author Vince Reuter
 */
class TestPartition extends AnyFunSuite, ScalaCheckPropertyChecks, should.Matchers:
    implicit override val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 1000)
    
    val simpleTypeArbitraries = Table("arb", Arbitrary[String], Arbitrary[Int], Arbitrary[Double])

    // Generate pair of number of subsets and set to partition such that the call to partition is legal.
    def genSmallLegalArgPair[A](g: Gen[A]): Gen[(Int, Set[A])] = for {
        n <- Gen.choose(1, 5) // Number of elements to partition
        k <- Gen.choose(1, n) // Request no more subsets than elements
        xs <- Gen.listOfN(n, g).suchThat(xs => xs.length === xs.toSet.size)
    } yield (k, xs.toSet)

    test("Every partition covers the input.") {
        forAll (genSmallLegalArgPair(arbitrary[String])) { 
            (n, xs) => partition(n, xs).forall(_.combineAll === xs) shouldBe true 
        }
    }

    test("Every partition consists of disjoint subsets.") {
        forAll (genSmallLegalArgPair(arbitrary[Int])) { 
            (n, xs) => partition(n, xs).forall(_.combinations(2).forall{
                case sub1 :: sub2 :: Nil => (sub1 & sub2).isEmpty
                case subs => throw new Exception(s"${subs.size}, 2, elements when taking pairs!")
            }) shouldBe true
        }
    }

    test("The number of valid partitions is as expected.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        forAll (genSmallLegalArgPair(arbitrary[Int]), minSuccessful(20)) { (k, xs) => 
            val n = xs.size
            // For expectation, see Theorem 5.2:
            // https://physics.byu.edu/faculty/berrondo/docs/physics-731/ballsinboxes.pdf
            // or "Distinguishable objects and indistinguishable boxes":
            // https://www.cs.wm.edu/~wm/CS243/ln11.pdf
            val numExp = (0 to (k - 1)).map{
                j => math.pow(-1, j) * (k `choose` j) * math.pow(k - j, n)
            }.sum / factorial(k)
            partition(k, xs).length shouldEqual numExp
        }
    }

    test("Requesting a non-integral number of subsets doesn't compile.") {
        assertCompiles{ "partition(2, (0 to 5).toSet))" }
        assertTypeError{ "partition(\"2\", (0 to 5).toSet))" }
        assertTypeError{ "partition(2.0, (0 to 5).toSet))" }
        assertTypeError{ "partition(2.toLong, (0 to 5).toSet))" }
        assertTypeError{ "partition(null, (0 to 5).toSet))" }
        assertTypeError{ "partition(Option.empty[Int], (0 to 5).toSet))" }
        assertTypeError{ "partition(List(2, 3), (0 to 5).toSet))" }
    }

    test("Requesting more subsets than elements is an IllegalArgumentException.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        def genArgs = for {
            xs <- arbitrary[Set[Int]]
            k <- Gen.choose(xs.size, Int.MaxValue)
        } yield (k, xs)
        forAll (genArgs) { (k, xs) => 
            val err = intercept[IllegalArgumentException]{ partition(k, xs) }
            val expMsg = s"requirement failed: Desired number of subsets exceeds number of elements: $k > ${xs.size}"
            err.getMessage shouldEqual expMsg
        }
    }

    test("Requesting a non-positive number of subsets is an IllegalArgumentException.") {
        forAll (Gen.choose(Int.MinValue, 0), arbitrary[Set[Int]]) { (k, xs) => 
            val err = intercept[IllegalArgumentException]{ partition(k, xs) }
            val expMsg = s"requirement failed: Desired number of subsets must be strictly postitive, not $k"
            err.getMessage shouldEqual expMsg
        }
    }

    // nCk, i.e. number of ways to choose k indistinguishable objects from n
    extension (n: Int)
        infix def choose(k: Int): Int = {
            require(k <= n, s"Cannot choose more items than available: $k > $n")
            factorial(n) / (factorial(k) * factorial(n - k))
        }

    private def factorial(n: Int): Int = {
        require(n >= 0, s"Cannot take factorial of a negative number: $n")
        (1 to n).product
    }

end TestPartition
