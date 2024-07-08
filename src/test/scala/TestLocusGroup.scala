package at.ac.oeaw.imba.gerlich.looptrace

import cats.data.NonEmptySet
import cats.syntax.all.*
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.LocusGroup
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/**
  * Tests for the [[at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.LocusGroup]] ADT.
  */
class TestLocusGroup extends AnyFunSuite with ScalaCheckPropertyChecks with should.Matchers:
    test("LocusGroup's locus times collection must be nonempty at the type level.") {
        assertTypeError{ "LocusGroup(ImagingTimepoint(NonnegativeInt(2)), Set(ImagingTimepoint(NonnegativeInt(1))))" }
        LocusGroup(ImagingTimepoint(NonnegativeInt(2)), NonEmptySet.one(ImagingTimepoint(NonnegativeInt(1)))).locusTimepoints 
        `shouldEqual` 
        NonEmptySet.one(ImagingTimepoint(NonnegativeInt(1)))
        val nonemptyLocusTimes = NonEmptySet.one(ImagingTimepoint(NonnegativeInt(1)))
    }

    test("LocusGroup cannot have regional time in the locus times.") {
        forAll (arbitrary[Boolean], Gen.choose(0, 9), Gen.nonEmptyListOf(Gen.choose(10, 99))) {
            (includeRegional, regional, loci) => 
                val rt = ImagingTimepoint.unsafe(regional)
                val initLoci = loci.map(ImagingTimepoint.unsafe)
                    .toNel
                    .getOrElse{ throw new Exception("Generated empty list of locus times!") }
                    .toNes
                if includeRegional then
                    val lts = initLoci.add(rt)
                    val error = intercept[IllegalArgumentException]{ LocusGroup(rt, lts) }
                    val expMsg = s"requirement failed: Regional time (${rt.get}) must not be in locus times!"
                    error.getMessage shouldEqual expMsg
                else
                    LocusGroup(rt, initLoci).locusTimepoints shouldEqual initLoci
        }
    }
end TestLocusGroup
