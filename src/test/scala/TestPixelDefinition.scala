package at.ac.oeaw.imba.gerlich.looptrace

import cats.* 
import cats.syntax.all.*
import pureconfig.*
import pureconfig.generic.semiauto.deriveReader
import squants.space.{ Length, LengthUnit, Nanometers }
import org.scalacheck.*
import org.scalactic.{ Equality, TolerantNumerics }
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import at.ac.oeaw.imba.gerlich.looptrace.space.{ PixelDefinition, Pixels3D }

/** Tests for the parsing of pixel definitions  */
class TestPixelDefinition extends AnyFunSuite, ScalaCheckPropertyChecks, should.Matchers:
    test("Basic example parses as expected."):
        import at.ac.oeaw.imba.gerlich.looptrace.configuration.instances.all.given

        given Arbitrary[Double] = Arbitrary{ Gen.choose(-10e6, 10e6) }
        given Shrink[Double] = Shrink.shrinkAny
        given Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-6)

        forAll(minSuccessful(10000)) { (x: Double, y: Double, z: Double) => 
            ConfigSource.string("{ x: 100 nm, y: 200 nm, z: 400 nm }").load[Pixels3D] match {
                case Left(errors) => fail(s"Errors: $errors")
                case Right(scaling) => 
                    (scaling.liftX(x) in Nanometers).value shouldEqual x * 100
                    (scaling.liftY(y) in Nanometers).value shouldEqual y * 200
                    (scaling.liftZ(z) in Nanometers).value shouldEqual z * 400
            }
        }
end TestPixelDefinition
