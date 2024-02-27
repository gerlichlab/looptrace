package at.ac.oeaw.imba.gerlich.looptrace

import scala.math.max
import cats.*
import cats.syntax.all.*
import mouse.boolean.*
import at.ac.oeaw.imba.gerlich.looptrace.space.*

/**
 * Tools for working with the quality control filtration and visualisation of locus-specific spots
 * 
 * @author Vince Reuter
 */
object LocusSpotQC:
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
    final case class DataRecord(
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
                )

        /** Helper to factor out the common structure in this comarison for each of the dimensions */
        private final def withinSigmaOfBound(sigma: Double, boxBound: PositiveReal)(p: Double): Either[Boolean, Boolean] = 
            (sigma > 0).either(0.0, sigma).mapBoth(s => p > s && p < boxBound - s)
    end DataRecord
    
    /** A bundle of the QC pass/fail components for individual rows/records supporting traces */
    final case class ResultRecord(withinRegion: Boolean, sufficientSNR: Boolean, denseXY: Boolean, denseZ: Boolean, inBoundsX: Boolean, inBoundsY: Boolean, inBoundsZ: Boolean):
        /** The individual true/false components indicating QC pass or fail. */
        final def components: Array[Boolean] = Array(withinRegion, sufficientSNR, denseXY, denseZ, inBoundsX, inBoundsY, inBoundsZ)
        /** Whether all of the QC check components in this instance indicate a pass */
        final def allPass: Boolean = components.all
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
     * @see [[DataRecord]]
     */
    final case class Signal(get: Double) extends AnyVal
    
    /** The background/baseline of a 3D Gaussian fit to a spot.
     * 
     * @see [[DataRecord]]
     */
    final case class Background(get: Double) extends AnyVal

    /** The standard deviation, in 'x' and 'y', of a 3D Gaussian fit to a spot.
     * 
     * @see [[DataRecord]]
     */
    final case class SigmaXY(get: PositiveReal) extends AnyVal
    
    /** Helpers for working with standard deviations of 3D Gaussian fit */
    object SigmaXY:
        /** Sort standard deviations by the underlying numerical value. */
        given sigmaXYOrder: Order[SigmaXY] = Order.by(_.get)
    end SigmaXY

    /** The standard deviation, in 'z', of a 3D Gaussian fit to a spot.
     * 
     * @see [[DataRecord]]
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
