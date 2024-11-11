package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ Failure, Success, Try }
import cats.* 
import cats.syntax.all.*
import pureconfig.*
import pureconfig.generic.semiauto.deriveReader
import squants.space.{ Length, LengthUnit, Microns, Nanometers }
import org.scalacheck.*
import org.scalactic.{ Equality, TolerantNumerics }
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import at.ac.oeaw.imba.gerlich.looptrace.space.{ LengthInNanometers, PixelDefinition, Pixels3D }
import at.ac.oeaw.imba.gerlich.looptrace.configuration.instances.all.given // Bring in the pureconfig.ConfigReader instances.

/** Tests for the parsing of pixel definitions  */
class TestPixelDefinition extends AnyFunSuite, ScalaCheckPropertyChecks, should.Matchers:

    private type MicroOrNano = Microns.type | Nanometers.type
    
    private def maxSize: Double = 1e6
    
    private def genMicroOrNano: Gen[MicroOrNano] = Gen.oneOf(Microns, Nanometers)
    
    private def genPlusMinusMillion: Gen[Double] = Gen.choose(-maxSize, maxSize)
    
    private type LengthParseInputs[A] = (A, LengthUnit)
    
    private def buildRawConfig[A: Show](
        x: LengthParseInputs[A], 
        y: LengthParseInputs[A], 
        z: LengthParseInputs[A],
    ): String = 
        given Show[LengthParseInputs[A]] = Show.show{ case (num, unit) => s"${num.show} ${unit.symbol}" }
        s"{ x: ${x.show}, y: ${y.show}, z: ${z.show} }"

    test("Basic examples parse as expected."):
        given Arbitrary[Double] = Arbitrary(genPlusMinusMillion)
        given Shrink[Double] = Shrink.shrinkAny
        given Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1 / maxSize)

        given Arbitrary[LengthUnit] = Arbitrary{ genMicroOrNano }

        forAll(minSuccessful(10000)) { (x: Double, y: Double, z: Double, unit: LengthUnit) => 
            val perX = 100
            val perY = 200
            val perZ = 400
            val rawConfigData = buildRawConfig(perX -> unit, perY -> unit, perZ -> unit)
            ConfigSource.string(rawConfigData).load[Pixels3D] match {
                case Left(errors) => fail(s"Errors: $errors")
                case Right(scaling) => 
                    (scaling.liftX(x) in unit).value shouldEqual x * perX
                    (scaling.liftY(y) in unit).value shouldEqual y * perY
                    (scaling.liftZ(z) in unit).value shouldEqual z * perZ
            }
        }

    test("Pixels3D parse requires the correct combination (x, y, z) of keys."):
        val legitKeys = Set("x", "y", "z")

        def genSubstitutions: Gen[List[(String, String)]] = 
            Gen.containerOf[Set, String](Gen.oneOf(legitKeys))
                .map(_.toList)
                .flatMap{ toSwapOut => 
                    Gen.listOfN(toSwapOut.length, Gen.alphaNumStr).map(toSwapOut.zip)
                }
        
        val MyUnit = Nanometers
        def genInputAndExpectation: Gen[(String, Boolean)] = 
            val base = buildRawConfig(100 -> MyUnit, 200 -> MyUnit, 300 -> MyUnit)
            genSubstitutions.map{ subs => 
                val updated = subs.foldLeft(base){ 
                    case (acc, (oldKey, newKey)) => acc.replace(oldKey ++ ":", newKey ++ ":") 
                }
                val expSuccess = 
                    val newKeys = legitKeys -- subs.map(_._1).toSet ++ subs.map(_._2).toSet
                    subs.isEmpty || legitKeys === newKeys
                updated -> expSuccess
            }

        given noShrink[A]: Shrink[A] = Shrink.shrinkAny // no shrinking whatsoever

        forAll (genInputAndExpectation, minSuccessful(1000)) { (rawConfigData, shouldSucceed) => 
            ConfigSource.string(rawConfigData).load[Pixels3D] match {
                case Left(errors) => 
                    if !shouldSucceed then succeed 
                    else fail(s"Expected succeess with $rawConfigData but failed: ${errors.prettyPrint}")
                case Right(_) => 
                    if shouldSucceed then succeed
                    else fail(s"Expected failure with $rawConfigData but succeeded")
            }
        }
    
    test("Pixels3D parse requires proper length units with strictly positive values."):
        val inputsAndExpectations = Table(
            ("rawConfigData", "shouldSucceed"), 
            ("{ x: 100 nm, y: 200 nm, z: 300 nm }", true), // good example
            ("{ x: 100 nm, y: 200, z: 300 nm }", false), // missing a units on y
            ("{ x: 100 nm, y: 200 nm, z: -300 nm }", false), // negative z
        )
        forAll (inputsAndExpectations) { (rawConfigData, shouldSucceed) => 
            ConfigSource.string(rawConfigData).load[Pixels3D] match {
                case Left(errors) => 
                    if !shouldSucceed then succeed 
                    else fail(s"Expected succeess with $rawConfigData but failed: ${errors.prettyPrint}")
                case Right(_) => 
                    if shouldSucceed then succeed
                    else fail(s"Expected failure with $rawConfigData but succeeded")
            }

        }

    test("With squants input, PixelDefinition.tryToDefine works exactly when the value is (strictly) positive."):
        def genLength = genMicroOrNano.flatMap{ buildLength => Gen.choose(-1e3, 1e3).map(buildLength.apply) }
        def genPixelCount: Gen[Int] = Gen.choose(0, 1e6.toInt)
        
        forAll (genLength, genPixelCount, minSuccessful(10000)) { (length, numPixels) =>
            PixelDefinition.tryToDefine(length) match {
                case Left(_) if length.value <= 0 => succeed
                case Left(msg) => fail(s"Pixel definition with length ${length} failed: $msg")
                case Right(_) if length.value <= 0 => fail(s"Pixel definition with length ${length} succeded")
                case Right(pxDef) => 
                    import PixelDefinition.syntax.lift
                    given Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1 / maxSize)

                    val obs: Length = (pxDef.lift(numPixels) in length.unit)
                    val exp: Length = 
                        val x = numPixels * length.value
                        val u = length.unit.symbol
                        Length(x, u)
                            .toEither
                            .leftMap{ e => s"Error bulding length from ($x, $u: ${e.getMessage})" }
                            .fold(msg => throw new Exception(msg), identity)
                    obs.value shouldEqual exp.value
                    obs.unit shouldEqual exp.unit
            }
        }
    
    test("PixelDefinition.tryToDefine is equivalent given a squants.space.Length input or a LengthInNanometers input."):
        def genLength: Gen[Length] = 
            given Arbitrary[Double] = Arbitrary{ Gen.choose(0.0, maxSize).suchThat(_ > 0) }
            genMicroOrNano.flatMap{ buildLength => Arbitrary.arbitrary[Double].map(buildLength.apply) }
        
        forAll (genLength, minSuccessful(10000)) { length => 
            val inNano: LengthInNanometers = LengthInNanometers.unsafeFromSquants(length)
            (PixelDefinition.tryToDefine(length), PixelDefinition.tryToDefine(inNano)) match {
                case (Left(msg1), Left(msg2)) => 
                    msg1 shouldEqual msg2
                case (Left(msg), Right(pxDef)) => 
                    fail(s"Squants input failed ($msg) while custom type (${inNano}) succeded, giving $pxDef")
                case (Right(pxDef), Left(msg)) => 
                    fail(s"Custom input failed failed ($msg) while squants input ($length) succeded, giving $pxDef")
                case (Right(def1), Right(def2)) => 
                    import PixelDefinition.syntax.lift
                    val fromLength = def1.lift(1)
                    val fromCustom = def2.lift(1)
                    (fromLength in length.unit) shouldEqual (fromCustom in length.unit)
            }
        }

    test("PixelDefinition.tryToDefine and PixelDefinition.unsafeDefine are equivalent."):
        given Shrink[Double] = Shrink.shrinkAny
        
        forAll (genPlusMinusMillion, genMicroOrNano, minSuccessful(10000)) { 
            (x: Double, u: LengthUnit) => 
                val length = Length(x, u.symbol)
                    .toEither
                    .leftMap{ e => s"Error bulding length from ($x, ${u.symbol}: ${e.getMessage})" }
                    .fold(msg => throw new Exception(msg), identity)
                (PixelDefinition.tryToDefine(length), Try{ PixelDefinition.unsafeDefine(length) }) match {
                    case (Left(msg), Failure(err)) =>
                        err.getMessage shouldEqual msg
                    case (Left(msg), Success(pxDef)) => 
                        fail(s"Unsafe version succeeded ($pxDef), but safe version failed with message: $msg")
                    case (Right(pxDef), Failure(err)) => 
                        fail(s"Safe version succeeded ($pxDef), but unsafe version failed with error: $err")
                    case (Right(defineFromSafe), Success(defineFromUnsafe)) => 
                        import PixelDefinition.syntax.lift
                        val fromSafe = defineFromSafe.lift(1)
                        val fromUnsafe = defineFromUnsafe.lift(1)
                        (fromSafe in length.unit).value shouldEqual (fromUnsafe in length.unit).value
                }
        }
end TestPixelDefinition
