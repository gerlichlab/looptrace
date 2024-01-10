package at.ac.oeaw.imba.gerlich.looptrace

import scala.collection.immutable.SortedSet
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
class TestCartesianProduct extends AnyFunSuite, ScalaCheckPropertyChecks, should.Matchers:
    implicit override val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 1000)

    /* For generating objects where type is of no importance */
    type Obj = String | Int | Double
    def genObj: Gen[Obj] = Gen.oneOf(arbitrary[String], arbitrary[Int], arbitrary[Double])
    
    test("Length-is-product-of-lengths: The length of cartesian product is the product of the length of the inputs.") {
        def genInputs: Gen[List[List[Obj]]] = Gen.resize(5, Gen.listOf(Gen.resize(5, Gen.listOf(genObj))))
        forAll (genInputs) { xss => cartesianProduct(xss).length shouldEqual xss.map(_.length).product }
    }

    test("As a corollary to length-is-product-of-length, any empty input collection results in empty output.") {
        def genInputs: Gen[List[List[Obj]]] = Gen.resize(5, Gen.listOf(Gen.resize(5, Gen.listOf(genObj))))
        forAll (genInputs.suchThat(_.exists(_.isEmpty))) { xss => cartesianProduct(xss).isEmpty shouldBe true }
    }

    test("When elements ARE unique within each input collection, each output collection IS unique.") {
        forAll (smallListGen(1, 5)(Gen.resize(5, arbitrary[SortedSet[Int]]))) { xss => 
            val prod = cartesianProduct(xss.map(_.toSeq))
            prod.toSet.size shouldEqual prod.length
        }
    }

    test("When there's an input collection with repeats, there are repeated output collections.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        def genInputs: Gen[List[List[Int]]] = 
            // Generate 2-5 sublists, each of 1-5 elements, with at least 1 sublist having repeats.
            smallListGen(2, 5)(smallListGen(1, 5)(Gen.choose(1, 10))).suchThat(_.exists(xs => xs.toSet.size =!= xs.length))
        forAll (genInputs) { xss => 
            val prod = cartesianProduct(xss.map(_.toSeq))
            prod.toSet.size < prod.length shouldBe true
        }
    }

    test("Each output collection has length equal to the number of input collections.") {
        def genObj: Gen[Obj] = Gen.oneOf(arbitrary[String], arbitrary[Int], arbitrary[Double])
        def genInputs: Gen[List[List[Obj]]] = Gen.resize(5, Gen.listOf(Gen.resize(5, Gen.listOf(genObj))))
        forAll (genInputs) { xss => cartesianProduct(xss).filter(_.length =!= xss.length) shouldEqual List() }
    }
    
    test("Each output collection takes 1 element from each input collection.") {
        given noShrink[A]: Shrink[A] = Shrink.shrinkAny[A]
        // Generate 2-5 sublists, each of 1-5 elements.
        def genInputs: Gen[List[List[Int]]] = smallListGen(2, 5)(smallListGen(1, 5)(Gen.choose(1, 10)))
        forAll (genInputs) { xss => 
            cartesianProduct(xss.map(_.toSeq)).forall(out => out.zip(xss).forall((x, src) => src `contains` x)) shouldBe true
        }
    }

    /** Generate a list of {@code A}, of relatively minimal size. */
    private def smallListGen[A](minSize: Int, maxSize: Int)(g: Gen[A]): Gen[List[A]] = Gen.choose(minSize, maxSize).flatMap(Gen.listOfN(_, g))
end TestCartesianProduct
