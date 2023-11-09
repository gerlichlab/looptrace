package at.ac.oeaw.imba.gerlich.looptrace

import cats.instances.string.*
import cats.syntax.eq.*
import cats.syntax.functor.*
import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

import at.ac.oeaw.imba.gerlich.looptrace.space.{ CoordinateSequence, Point3D }
import at.ac.oeaw.imba.gerlich.looptrace.PartitionIndexedDriftCorrectionRois.{ XColumn, YColumn, ZColumn, ParserConfig }

/** Types and helpers for testing partitioning of regions of interest (ROIs) */
trait PartitionRoisSuite extends LooptraceSuite:
    
    /** Randomized, guaranteed-valid parser configuration (no column name collisions) */
    given parserConfigArbitrary: Arbitrary[ParserConfig] = Arbitrary(genParserConfig)
    
    /** Arbitrary point, index, and usability flag lifted into detected ROI data type value */
    given detectedRoiArbitrary(using arbComponents: Arbitrary[(RoiIndex, Point3D, Boolean)]): Arbitrary[DetectedRoi] = 
        arbComponents.fmap(DetectedRoi.apply.tupled)
    
    /** Generator for detected ROI, fixing the usability flag as given */
    def genDetectedRoiFixedUse = (p: Boolean) => arbitrary[DetectedRoi].map(_.copy(isUsable = p))

    /** Arbitrary point, index, and lifted into shifting ROI data type value */
    given shiftingRoiArbitrary: Arbitrary[RoiForShifting] = Arbitrary{ genSelectedRoi(RoiForShifting.apply) }
    
    /** Arbitrary point, index, and lifted into accuracy ROI data type value */
    given accuracyRoiArbitrary: Arbitrary[RoiForAccuracy] = Arbitrary{ genSelectedRoi(RoiForAccuracy.apply) }
    
    given xColArb: Arbitrary[XColumn] = Arbitrary{ Gen.alphaNumStr.suchThat(_.nonEmpty).map(XColumn.apply) }
    given yColArb: Arbitrary[YColumn] = Arbitrary{ Gen.alphaNumStr.suchThat(_.nonEmpty).map(YColumn.apply) }
    given zColArb: Arbitrary[ZColumn] = Arbitrary{ Gen.alphaNumStr.suchThat(_.nonEmpty).map(ZColumn.apply) }

    /**
     * Generate a legal/valid parsing configuration for bead ROIs partition
     * 
     * This function ensures that there's no column name collision for the parser configuration, 
     * and that the values of each field are appropriately typed.
     * 
     * @return A generator of valid parser configurations
     */
    def genParserConfig: Gen[ParserConfig] = for {
        x <- arbitrary[XColumn]
        y <- Gen.alphaNumStr.suchThat(_ =!= x.get).map(YColumn.apply)
        z <- Gen.alphaNumStr.suchThat(s => s =!= x.get && s =!= y.get).map(ZColumn.apply)
        qc <- Gen.alphaNumStr.suchThat(s => s =!= x.get && s =!= y.get && s =!= z.get)
        coordseq <- arbitrary[CoordinateSequence]
    } yield ParserConfig(x, y, z, qcCol = qc, coordinateSequence = coordseq)

    /** Generate a point and an index, then lift these arguments into a (isomorphic) case class instance. */
    def genSelectedRoi[R <: SelectedRoi](build: (RoiIndex, Point3D) => R): Gen[R] = for {
        i <- arbitrary[RoiIndex]
        p <- arbitrary[Point3D]
    } yield build(i, p)
