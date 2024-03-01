package at.ac.oeaw.imba.gerlich.looptrace

import scala.language.adhocExtensions // to extend ujson.Value.InvalidData
import scala.math.max
import cats.*
import cats.data.*
import cats.data.Validated.{ Invalid, Valid }
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*

import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.*

/**
 * Tools for working with the quality control filtration and visualisation of locus-specific spots
 * 
 * @author Vince Reuter
 */
object LocusSpotQC:

    object PointCodec:
        private[LocusSpotQC] def toJsonObject(p: Point3D): ujson.Obj = ujson.Obj(
            "x" -> ujson.Num(p.x.get),
            "y" -> ujson.Num(p.y.get),
            "z" -> ujson.Num(p.z.get),
        )

        private[LocusSpotQC] def fromJson(json: ujson.Value): ErrMsgsOr[Point3D] = ???

    /**
      * Bundle of data that uniquely identifies a spot and gives its coordinates and QC result.
      *
      * @param identifier The context with which to identify the spot within an experiment
      * @param point The coordinates of the centroid of the 3D Gaussian fit to the pixels
      * @param result The results of the QC filters
      */
    final case class OutputRecord(identifier: SpotIdentifier, point: Point3D, qcResult: ResultRecord):
        final def passesQC: Boolean = qcResult.allPass
        final def canBeDisplayed: Boolean = qcResult.canBeDisplayed
        final def traceId: TraceId = identifier.traceId
        final def time: Timepoint = identifier.locusId.get
        final def z: ZCoordinate = point.z
        final def y: YCoordinate = point.y
        final def x: XCoordinate = point.x

    object OutputRecord:
        /** JSON codec */
        def rwForOutputRecord: ReadWriter[OutputRecord] = readwriter[ujson.Value].bimap(
            record => ujson.Obj(
                "identifier" -> SpotIdentifier.toJsonObject(record.identifier), 
                "point" -> PointCodec.toJsonObject(record.point), 
                "qcResult" -> ResultRecord.toJsonObject(record.qcResult),
            ), 
            json => ???
        )

    /**
      * Bundle of data with which to uniquely identify a locus-specific spot in an experiment
      *
      * @param position Field of view, 0-based index
      * @param regionId Wrapper around the timepoint at which the spot's associated region was imaged
      * @param traceId 0-based index identifying the trace, ideally uniquely within the experiment but perhaps not
      * @param locusId Wrapper around the timepoint at which the spot was imaged
      */
    final case class SpotIdentifier(position: PositionIndex, regionId: RegionId, traceId: TraceId, locusId: LocusId)
    
    /** Helpers for working with identifiers of locus-specific spots */
    object SpotIdentifier:
        /**
          * Error subtype for when something goes wrong decoding an instance from JSON
          *
          * @param messages Why decoding was not possible
          * @param json The data on which the decoding attempt was made
          */
        final class DecodingError(messages: NonEmptyList[String], json: ujson.Value) 
            extends ujson.Value.InvalidData(json, s"Error(s) decoding locus spot identifier: ${messages.mkString_("; ")}")
        
        private[LocusSpotQC] def toJsonObject(spotId: SpotIdentifier): ujson.Obj = ujson.Obj(
            "position" -> ujson.Num(spotId.position.get),
            "regionId" -> ujson.Num(spotId.regionId.get.get),
            "traceId" -> ujson.Num(spotId.traceId.get), 
            "locusId" -> ujson.Num(spotId.locusId.get.get), 
        )

        private[looptrace] def fromJson(json: ujson.Value): ErrMsgsOr[SpotIdentifier] = ???

        /** A JSON codec which unwraps the components and maps field names to the refined values */
        def rwForSpotIdentifier: ReadWriter[SpotIdentifier] = readwriter[ujson.Value].bimap(
            toJsonObject,
            json => 
                val posIdNel = safeExtractE("position", safeParseInt >>> PositionIndex.fromInt)(json)
                val regIdNel = safeExtractE("regionId", safeParseInt >>> RegionId.fromInt)(json)
                val traceIdNel = safeExtractE("traceId", safeParseInt >>> TraceId.fromInt)(json)
                val locusIdNel = safeExtractE("locusId", safeParseInt >>> LocusId.fromInt)(json)
                (posIdNel, regIdNel, traceIdNel, locusIdNel).mapN(SpotIdentifier.apply) match {
                    case Invalid(messages) => throw new DecodingError(messages, json)
                    case Valid(spotId) => spotId
                }
        )
    end SpotIdentifier

    /**
      * Data, from a single looptrace tracing record, that's associated with quality control of a locus-specific spot
      *
      * @param bounds Upper bound on the pixel coordinate in each dimension for the box associated with this record
      * @param centroid The center of the 3D Gaussian fit to (possibly transformed) pixel intensity values in the `box`
      * @param distanceToRegion The (Euclidean) distance between `centroid` of this spot and the centroid of the 3D Gaussian 
      *     fit to the regional barcode spot with which the locus-specific spot is associated
      * @param signal Measure of the peak intensity of the 3D Gaussian fit to this spot's pixel values; 
      *     TODO: see more about what exactly this is 
      * @param background Measure of the baseline pixel intensity in this region; TODO: see more about what exactly this is
      * @param sigmaXY The standard deviation, in the 'xy' plane, of the 3D Gaussian fit to this spot
      * @param sigmaZ The standard deviation, in 'z', of the 3D Gaussian fit to this spot
      */
    final case class InputRecord(
        bounds: BoxUpperBounds,
        centroid: Point3D,
        distanceToRegion: DistanceToRegion, 
        signal: Signal, 
        background: Background, 
        sigmaXY: Double, 
        sigmaZ: Double,
        ):
        
        /** 'x'-coordinate of the `centroid` */
        final def x: XCoordinate = centroid.x
        
        /** 'y'-coordinate of the `centroid` */
        final def y: YCoordinate = centroid.y
        
        /** 'z'-coordinate of the `centroid` */
        final def z: ZCoordinate = centroid.z
        
        /** Whether this record's spot's centroid is within the given distance of the center of the associated regional spot */
        final def isCloseEnoughToRegion(maxDist: DistanceToRegion): Boolean = distanceToRegion < maxDist
        
        /** Whether the 3D Gaussian fit to this spot is less diffuse than the given standard deviation, in 'x' and in 'y' */
        final def isConcentratedXY(maxSigma: SigmaXY): Boolean = sigmaXY > 0 && sigmaXY < maxSigma.get
        
        /** Whether the 3D Gaussian fit to this spot is less diffuse than the given standard deviation, in 'z */
        final def isConcentratedZ(maxSigma: SigmaZ): Boolean = sigmaZ > 0 && sigmaZ < maxSigma.get
        
        /**
         * Whether this data instance's ratio of signal-to-noise is sufficiently high to pass quality control
         * 
         * @param minSNR The minimum ratio of signal-to-noise needed to "pass" quality control
         * @return Whether the `signal` is greater than the product of `minSNR` and `background`
         */
        final def hasSufficientSNR(minSNR: SignalToNoise): Boolean = signal.get > minSNR.get * background.get
        
        /**
         * Use the criteria provided to determine the QC result, including per-component values, of this record.
         * 
         * @param maxDistanceToRegion The upper bound permitted on distance between a locus spot's centroid and the centroid 
         *     of the regional spot with which it's associated
         * @param minSNR The minimum signal-to-noise ratio required
         * @param maxSigmaXY The upper limit of standard deviation in 'x' and 'y' to tolerate
         * @param maxSigmaZ The upper limit of standard deviation in 'z' to tolerate
         * @return An instance wrapping the component-wise QC pass/fail flag and capable of evaluating the overall QC state of the record
         */
        final def toQCResult(maxDistanceToRegion: DistanceToRegion, minSNR: SignalToNoise, maxSigmaXY: SigmaXY, maxSigmaZ: SigmaZ): ResultRecord = 
            ResultRecord(
                withinRegion = isCloseEnoughToRegion(maxDistanceToRegion), 
                sufficientSNR = hasSufficientSNR(minSNR), 
                denseXY = isConcentratedXY(maxSigmaXY), 
                denseZ = isConcentratedZ(maxSigmaZ), 
                inBoundsX = withinSigmaOfBound(sigma = sigmaXY, boxBound = bounds.x.get)(x.get).merge, 
                inBoundsY = withinSigmaOfBound(sigma = sigmaXY, boxBound = bounds.y.get)(y.get).merge, 
                inBoundsZ = withinSigmaOfBound(sigma = sigmaZ, boxBound = bounds.z.get)(z.get).merge, 
                canBeDisplayed = (
                    centroid.z.get > 0 && 
                    centroid.z.get < bounds.z.get && 
                    centroid.y.get > 0 && 
                    centroid.y.get < bounds.y.get && 
                    centroid.x.get > 0 && 
                    centroid.x.get < bounds.x.get
                    )
                )

        /**
         * Helper to factor out the common structure in this comarison for each of the dimensions, used along each of the 3 coordinate axes
         *
         * @param sigma The standard deviation, along a particular axis, of a 3D Gaussian fit to the locus spot
         * @param boxBound The (upper) bound in a particular dimension of the bounding box around a locus-specific spot
         * @param p The coordinate of a locus spot's center in a particular dimension
         * @return Whether the given point is sufficiently "inside" (more than 1 `sigma`) the interval of the locus spot's 
         *     bounding box, with the interval being in a particular one of the three dimensions
         */
        private final def withinSigmaOfBound(sigma: Double, boxBound: PositiveReal)(p: Double): Either[Boolean, Boolean] = 
            (sigma > 0).either(0.0, sigma).mapBoth(s => p > s && p < boxBound - s)
    end InputRecord
    
    /** A bundle of the QC pass/fail components for individual rows/records supporting traces
      *
      * @param withinRegion Whether a locus spot's center was sufficiently close to the center of its regional spot
      * @param sufficientSNR Whether the 3D Gaussian fit to a locus spot's pixels had sufficient signal-to-noise ratio
      * @param denseXY Whether the 3D Gaussian fit to a locus spot has a standard deviation of no more than a certain value in 'x' and 'y'
      * @param denseZ Whether the 3D Gaussian fit to a locus spot has a standard deviation of no more than a certain value in 'z'
      * @param inBoundsX Whether the center of a 3D Gaussian fit to a locus spot is more than 1 standar deviation away from both of the 'x' bounds 
      *     of the spot's bounding box
      * @param inBoundsY Whether the center of a 3D Gaussian fit to a locus spot is more than 1 standar deviation away from both of the 'y' bounds 
      *     of the spot's bounding box
      * @param inBoundsZ Whether the center of a 3D Gaussian fit to a locus spot is more than 1 standar deviation away from both of the 'z' bounds 
      *     of the spot's bounding box
      * @param canBeDisplayed Whether the locus spot's QC label will be able to be displayed in `napari`
      */
    final case class ResultRecord(
        withinRegion: Boolean, 
        sufficientSNR: Boolean, 
        denseXY: Boolean, 
        denseZ: Boolean, 
        inBoundsX: Boolean, 
        inBoundsY: Boolean, 
        inBoundsZ: Boolean,
        canBeDisplayed: Boolean,
        ):
        /** The individual true/false components indicating QC pass or fail. */
        final def components: Array[Boolean] = Array(withinRegion, sufficientSNR, denseXY, denseZ, inBoundsX, inBoundsY, inBoundsZ)
        /** Whether all of the QC check components in this instance indicate a pass */
        final def allPass: Boolean = components.all
    end ResultRecord

    /** Helpers for working with a bundle of QC result components for a single locus-specific spot */
    object ResultRecord:
        /** Write the record to a JSON object. */
        private[LocusSpotQC] def toJsonObject(r: ResultRecord): ujson.Obj = ujson.Obj(
            "withinRegion" -> ujson.Bool(r.withinRegion),
            "sufficientSNR" -> ujson.Bool(r.sufficientSNR), 
            "denseXY" -> ujson.Bool(r.denseXY), 
            "denseZ" -> ujson.Bool(r.denseZ), 
            "inBoundsX" -> ujson.Bool(r.inBoundsX),
            "inBoundsY" -> ujson.Bool(r.inBoundsY),
            "inBoundsZ" -> ujson.Bool(r.inBoundsZ),
            "canBeDisplayed" -> ujson.Bool(r.canBeDisplayed),
        )

        private[LocusSpotQC] def fromJson(json: ujson.Value): ErrMsgsOr[ResultRecord] = ???
    end ResultRecord

    /** The (Euclidean)  distance between a locus-specific spot's center and the center of its associated regional spot */
    final case class DistanceToRegion(get: NonnegativeReal) extends AnyVal

    /** Helpers for working with distances between centers of locus-specific and regional spots */
    object DistanceToRegion:
        /** Sort distances by the underlying numerical value. */
        given distToRegionOrd: Order[DistanceToRegion] = Order.by(_.get)
    end DistanceToRegion

    /** A signal-to-noise ratio */
    final case class SignalToNoise(get: PositiveReal) extends AnyVal

    /** Helpers for working with signal-to-noise ratios */
    object SignalToNoise:
        /** Sort ratios by the underlying numerical value. */
        given snrOrder: Order[SignalToNoise] = Order.by(_.get)
    end SignalToNoise

    /** The amplitude/peak of a 3D Gaussian fit to a spot.
     * 
     * @see [[InputRecord]]
     */
    final case class Signal(get: Double) extends AnyVal
    
    /** The background/baseline of a 3D Gaussian fit to a spot.
     * 
     * @see [[InputRecord]]
     */
    final case class Background(get: Double) extends AnyVal

    /** The standard deviation, in 'x' and 'y', of a 3D Gaussian fit to a spot.
     * 
     * @see [[InputRecord]]
     */
    final case class SigmaXY(get: PositiveReal) extends AnyVal
    
    /** Helpers for working with standard deviations of 3D Gaussian fit */
    object SigmaXY:
        /** Sort standard deviations by the underlying numerical value. */
        given sigmaXYOrder: Order[SigmaXY] = Order.by(_.get)
    end SigmaXY

    /** The standard deviation, in 'z', of a 3D Gaussian fit to a spot.
     * 
     * @see [[InputRecord]]
     */
    final case class SigmaZ(get: PositiveReal) extends AnyVal

    /** Helpers for working with standard deviations of 3D Gaussian fit */
    object SigmaZ:
        /** Sort ratios by the underlying numerical value. */
        given sigmaZOrder: Order[SigmaZ] = Order.by(_.get)
    end SigmaZ

    /** The side length, in 'z', of the bounding region of a spot */
    final case class BoxBoundZ(get: PositiveReal) extends AnyVal
    
    /** Helpers for working with the rectangular prism in which a spot is analysed */
    object BoxBoundZ:
        /** Sort box size by underlying numerical value. */
        given boxZOrder: Order[BoxBoundZ] = Order.by(_.get)
    end BoxBoundZ

    /** The side length, in 'y', of the bounding region of a spot */
    final case class BoxBoundY(get: PositiveReal) extends AnyVal
    
    /** Helpers for working with the rectangular prism in which a spot is analysed */
    object BoxBoundY:
        /** Sort box size by underlying numerical value. */
        given boxZOrder: Order[BoxBoundY] = Order.by(_.get)
    end BoxBoundY

    /** The side length, in 'x', of the bounding region of a spot */
    final case class BoxBoundX(get: PositiveReal) extends AnyVal
    
    /** Helpers for working with the rectangular prism in which a spot is analysed */
    object BoxBoundX:
        /** Sort box size by underlying numerical value. */
        given boxZOrder: Order[BoxBoundX] = Order.by(_.get)
    end BoxBoundX

    final case class BoxUpperBounds(x: BoxBoundX, y: BoxBoundY, z: BoxBoundZ)

end LocusSpotQC
