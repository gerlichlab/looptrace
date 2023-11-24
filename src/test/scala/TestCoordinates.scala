package at.ac.oeaw.imba.gerlich.looptrace.space

import scala.util.{ NotGiven, Random }
import cats.syntax.order.*
import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should

import at.ac.oeaw.imba.gerlich.looptrace.{ sortByCats, ScalacheckSuite }

/** Tests for the geometric coordinate abstractions */
class TestCoordinates extends AnyFunSuite, ScalacheckSuite, should.Matchers:
    
    test("Ordering coordinates works.") {
        enum CoordinateKey:
            case X, Y, Z

        given arbCoordinateKey: Arbitrary[CoordinateKey] = 
            Arbitrary{ Gen.oneOf(CoordinateKey.X, CoordinateKey.Y, CoordinateKey.Z) }

        forAll (Gen.zip(arbitrary[CoordinateKey], arbitrary[List[Double]])) { 
            case (CoordinateKey.X, xs) => assertOrder(xs, XCoordinate.apply)
            case (CoordinateKey.Y, ys) => assertOrder(ys, YCoordinate.apply)
            case (CoordinateKey.Z, zs) => assertOrder(zs, ZCoordinate.apply)
        }
    }
    
    test("Coordinates to order cannot be of different types.") {
        /* Positive pretests */
        assertCompiles("XCoordinate(1.0) < XCoordinate(2.0)")
        assertCompiles("YCoordinate(-1.0) >= YCoordinate(2.0)")
        assertCompiles("ZCoordinate(1.0) <= ZCoordinate(2.0)")
        
        /* NB: don't check triple-equals / cats.Eq syntax here; may conflate with scalatest/scalacheck. */
        assertTypeError("XCoordinate(1.0) < YCoordinate(2.0)")
        assertTypeError("YCoordinate(-1.0) >= ZCoordinate(2.0)")
        assertTypeError("ZCoordinate(1.0) <= XCoordinate(2.0)")
    }
    
    test("Generic coordinates cannot be ordered, even if the same underlying type") {
        /* Positive pretests */
        assertCompiles("XCoordinate(1.0): Coordinate") // x1
        assertCompiles("XCoordinate(2.0): Coordinate") // x2
        assertCompiles("YCoordinate(1): Coordinate") // y1
        assertCompiles("YCoordinate(2): Coordinate") // y2
        assertCompiles("ZCoordinate(2.0): Coordinate") // z1
        assertCompiles("ZCoordinate(3): Coordinate") // z2

        /* Failures for runtime-X */
        assertTypeError{ "(XCoordinate(1.0): Coordinate) < (XCoordinate(2.0): Coordinate)" }
        assertTypeError{ "(XCoordinate(1.0): Coordinate) <= (XCoordinate(2.0): Coordinate)" }
        assertTypeError{ "(XCoordinate(1.0): Coordinate) > (XCoordinate(2.0): Coordinate)" }
        assertTypeError{ "(XCoordinate(1.0): Coordinate) >= (XCoordinate(2.0): Coordinate)" }

        /* Failures for runtime-Y */
        assertTypeError{ "(YCoordinate(1): Coordinate) < (YCoordinate(2): Coordinate)"}
        assertTypeError{ "(YCoordinate(1): Coordinate) <= (YCoordinate(2): Coordinate)"}
        assertTypeError{ "(YCoordinate(1): Coordinate) > (YCoordinate(2): Coordinate)"}
        assertTypeError{ "(YCoordinate(1): Coordinate) >= (YCoordinate(2): Coordinate)"}

        /* Failures for runtime-Z */
        assertTypeError{ "(ZCoordinate(2.0): Coordinate) < (ZCoordinate(3): Coordinate)" }
        assertTypeError{ "(ZCoordinate(2.0): Coordinate) <= (ZCoordinate(3): Coordinate)" }
        assertTypeError{ "(ZCoordinate(2.0): Coordinate) > (ZCoordinate(3): Coordinate)" }
        assertTypeError{ "(ZCoordinate(2.0): Coordinate) >= (ZCoordinate(3): Coordinate)" }
    }

    test("Coordinate cannot be extended.") {
        // TODO: assert this message -- "Cannot extend sealed trait Coordinate in a different source file"
        assertTypeError("new Coordinate{ def get = 1.0 }")
        assertTypeError("new Coordinate{ def get = 1 }")
    }

    def assertOrder[C <: Coordinate : [C] =>> NotGiven[C =:= Coordinate]](values: List[Double], build: Double => C) = {
        val orderedFirst = values.sorted.map(build)
        val builtFirst = Random.shuffle(values).map(build).sortByCats
        orderedFirst shouldEqual builtFirst
    }

end TestCoordinates
