package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ NotGiven, Try }
import cats.*
import cats.syntax.all.*
import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary

import com.github.tototoshi.csv.*
import io.github.iltotore.iron.scalacheck.all.given // for generation in compliance with ValidTraceGroupName constraint

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.geometry.{BoundingBox, Centroid, EuclideanDistance}
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.roi.DetectedSpot
import at.ac.oeaw.imba.gerlich.gerlib.testing.instances.{ GeometricInstances, ImagingInstances, CatsScalacheckInstances }
import at.ac.oeaw.imba.gerlich.gerlib.testing.syntax.SyntaxForScalacheck

import at.ac.oeaw.imba.gerlich.looptrace.roi.{ 
    DetectedSpotRoi, 
    MergedRoiRecord, 
}
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.{
    IndexedDetectedSpot, 
    PostMergeRoi,
}
import at.ac.oeaw.imba.gerlich.looptrace.space.{
    BoundingBox as BB,
    Coordinate, 
    Point3D,
    XCoordinate, 
    YCoordinate, 
    ZCoordinate,
}
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Base trait for tests in looptrace */
trait LooptraceSuite extends GenericSuite, GeometricInstances, ImagingInstances, CatsScalacheckInstances, SyntaxForScalacheck:

    /************************/
    /* Givens ("implicits") */
    /************************/
    given arbitraryForDelimiter: Arbitrary[Delimiter] = Arbitrary{ Gen.oneOf(Delimiter.CommaSeparator, Delimiter.TabSeparator) }

    given (arbT: Arbitrary[NonnegativeReal]) => Arbitrary[EuclideanDistance.Threshold] = 
        arbT.map(EuclideanDistance.Threshold.apply)

    given (arbTime: Arbitrary[ImagingTimepoint]) => Arbitrary[LocusId] = arbTime.map(LocusId.apply)

    given Arbitrary[OneBasedFourDigitPositionName] = 
        val rawString = (p: Int) => "P" ++ "%04d".format(p)
        Arbitrary(Gen.choose(1, 9999).map(rawString andThen unsafeLiftStringToOneBasedFourDigitPositionName))

    given (arbName: Arbitrary[String]) => Arbitrary[ProbeName] = arbName.suchThat(_.nonEmpty).map(ProbeName.apply)

    given (arbTime: Arbitrary[ImagingTimepoint]) => Arbitrary[RegionId] = arbTime.map(RegionId.apply)

    given (Arbitrary[Int]) => Arbitrary[RoiIndex] = 
        Arbitrary{ Gen.choose(0, Int.MaxValue).map(RoiIndex.unsafe) }

    given Arbitrary[TraceGroupId] = summon[Arbitrary[ValidTraceGroupName]].map(TraceGroupId.apply)

    given (Arbitrary[TraceGroupId]) => Arbitrary[TraceGroupMaybe] = 
        summon[Arbitrary[Option[TraceGroupId]]].map(TraceGroupMaybe.apply)

    given (arbName: Arbitrary[String], arbTime: Arbitrary[ImagingTimepoint]) => Arbitrary[BlankImagingRound] = 
        (arbName, arbTime).mapN(BlankImagingRound.apply)

    given (
        arbName: Arbitrary[String], 
        arbTime: Arbitrary[ImagingTimepoint], 
        arbProbe: Arbitrary[ProbeName]
    ) => Arbitrary[RegionalImagingRound] = (arbName, arbTime, arbProbe).mapN(RegionalImagingRound.apply)

    given ( 
        arbName: Arbitrary[String], 
        arbTime: Arbitrary[ImagingTimepoint], 
        arbProbe: Arbitrary[ProbeName], 
        arbRepeat: Arbitrary[PositiveInt]
    ) => Arbitrary[LocusImagingRound] = 
        val arbRepOpt = Gen.option(arbitrary(arbRepeat)).toArbitrary
        (arbName, arbTime, arbProbe, arbRepOpt).mapN(LocusImagingRound.apply)

    given (
        arbSpot: Arbitrary[DetectedSpot[Double]], 
        arbBox: Arbitrary[BoundingBox[Double]],
    ) => Arbitrary[DetectedSpotRoi] = (arbSpot, arbBox).mapN(DetectedSpotRoi.apply)

    given (
        arbIndex: Arbitrary[RoiIndex],
        arbContext: Arbitrary[ImagingContext], 
        arbCentroid: Arbitrary[Centroid[Double]], 
        arbBox: Arbitrary[BB], 
    ) => Arbitrary[IndexedDetectedSpot] = 
        (arbIndex, arbContext, arbCentroid, arbBox).mapN(IndexedDetectedSpot.apply)

    given ( 
        Arbitrary[IndexedDetectedSpot], 
        Arbitrary[MergedRoiRecord],
    ) => Arbitrary[PostMergeRoi] = 
        Arbitrary.oneOf[IndexedDetectedSpot, MergedRoiRecord]
    
    /************************
     * Other definitions
     ***********************/
    protected def genNonNegReal(limit: NonnegativeReal): Gen[NonnegativeReal] = Gen.choose(0.0, limit).map(NonnegativeReal.unsafe)
    
    protected def genPosReal(limit: PositiveReal): Gen[PositiveReal] = Gen.choose(0.0, limit).suchThat(_ > 0).map(PositiveReal.unsafe)

    protected def unsafeLiftStringToOneBasedFourDigitPositionName(s: String): OneBasedFourDigitPositionName =
        OneBasedFourDigitPositionName
            .fromString(false)(s)
            .fold(msg => throw new IllegalArgumentException(s"Failed to refine value ($s): $msg"), identity)

    /**
      * Read given file as CSV with header, and handle resource safety.
      *
      * @param f The path to the file to read as CSV
      * @return Either a [[scala.util.Left]]-wrapped exception or a [[scala.util.Right]]-wrapped pair of columns and list of row records
      */
    protected def safeReadAllWithOrderedHeaders(f: os.Path): Either[Throwable, (List[String], List[Map[String, String]])] = for
        reader <- Try{ CSVReader.open(f.toIO) }.toEither
        result <- Try{ reader.allWithOrderedHeaders() }.toEither
        _ = reader.close()
    yield result
end LooptraceSuite