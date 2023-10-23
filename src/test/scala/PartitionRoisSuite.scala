package at.ac.oeaw.imba.gerlich.looptrace

import cats.instances.string.*
import cats.syntax.eq.*
import cats.syntax.functor.*
import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

import at.ac.oeaw.imba.gerlich.looptrace.space.{ CoordinateSequence, Point3D }
import at.ac.oeaw.imba.gerlich.looptrace.PartitionIndexedPoints.{ XColumn, YColumn, ZColumn, ParserConfig }

/** Types and helpers for testing partitioning of regions of interest (ROIs) */
trait PartitionRoisSuite extends LooptraceSuite:
    given parserConfigArbitrary: Arbitrary[ParserConfig] = Arbitrary(genParserConfig)
    given detectedRoiArbitrary(using arbComponents: Arbitrary[(RoiIndex, Point3D, Boolean)]): Arbitrary[DetectedRoi] = 
        arbComponents.fmap(DetectedRoi.apply.tupled)
    given shiftingRoiArbitrary: Arbitrary[RoiForShifting] = Arbitrary{ genSelectedRoi(RoiForShifting.apply) }
    given accuracyRoiArbitrary: Arbitrary[RoiForAccuracy] = Arbitrary{ genSelectedRoi(RoiForAccuracy.apply) }
    
    def genParserConfig: Gen[ParserConfig] = for {
        x <- arbitrary[String].map(XColumn.apply)
        y <- arbitrary[String].suchThat(_ =!= x.get).map(YColumn.apply)
        z <- arbitrary[String].suchThat(s => s =!= x.get && s =!= y.get).map(ZColumn.apply)
        qc <- arbitrary[String].suchThat(s => s =!= x.get && s =!= y.get && s =!= z.get)
        coordseq <- arbitrary[CoordinateSequence]
    } yield ParserConfig(x, y, z, qcCol = qc, coordinateSequence = coordseq)

    def genSelectedRoi[R <: SelectedRoi](build: (RoiIndex, Point3D) => R): Gen[R] = for {
        i <- arbitrary[RoiIndex]
        p <- arbitrary[Point3D]
    } yield build(i, p)
