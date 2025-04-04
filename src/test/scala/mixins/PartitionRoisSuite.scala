package at.ac.oeaw.imba.gerlich.looptrace

import cats.instances.string.*
import cats.syntax.eq.*
import cats.syntax.functor.*
import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

import at.ac.oeaw.imba.gerlich.looptrace.space.Point3D
import at.ac.oeaw.imba.gerlich.looptrace.PartitionIndexedDriftCorrectionBeadRois.{ XColumn, YColumn, ZColumn, ParserConfig }

/** Types and helpers for testing partitioning of regions of interest (ROIs) */
trait PartitionRoisSuite extends LooptraceSuite:

    /** Arbitrary point, index, and usability flag lifted into detected ROI data type value */
    given (arbComponents: Arbitrary[(RoiIndex, Point3D)]) => Arbitrary[FiducialBead] = 
        arbComponents.fmap(FiducialBead.apply.tupled)
    
    /** Arbitrary point, index, and lifted into shifting ROI data type value */
    given arbitraryForRoiForShifting: Arbitrary[RoiForShifting] = Arbitrary{ genSelectedRoi(RoiForShifting.apply) }
    
    /** Arbitrary point, index, and lifted into accuracy ROI data type value */
    given arbitraryForRoiForAccuracy: Arbitrary[RoiForAccuracy] = Arbitrary{ genSelectedRoi(RoiForAccuracy.apply) }
    
    /* Arbitrary instances for (z, y, x) columns to parse */
    given xColArb: Arbitrary[XColumn] = Arbitrary{ Gen.alphaNumStr.suchThat(_.nonEmpty).map(XColumn.apply) }
    given yColArb: Arbitrary[YColumn] = Arbitrary{ Gen.alphaNumStr.suchThat(_.nonEmpty).map(YColumn.apply) }
    given zColArb: Arbitrary[ZColumn] = Arbitrary{ Gen.alphaNumStr.suchThat(_.nonEmpty).map(ZColumn.apply) }

    /** Generate a point and an index, then lift these arguments into a (isomorphic) case class instance. */
    private def genSelectedRoi[R <: SelectedRoi](build: (RoiIndex, Point3D) => R): Gen[R] = for
        i <- arbitrary[RoiIndex]
        p <- arbitrary[Point3D]
    yield build(i, p)
