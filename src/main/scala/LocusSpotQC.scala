package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
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
      * @param box The representation of the rectangular prism in which the particular spot was detected (pixel coordinates)
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
        box: (BoxSizeZ, BoxSizeY, BoxSizeX),
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
        /**
         * Whether this data instance's ratio of signal-to-noise is sufficiently high to pass quality control
         * 
         * @param minSNR The minimum ratio of signal-to-noise needed to "pass" quality control
         * @return Whether the `signal` is greater than the product of `minSNR` and `background`
         */
        final def passesSNR(minSNR: SignalToNoise): Boolean = signal.get > minSNR.get * background.get
    end DataRecord
    
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
    final case class BoxSizeZ(get: PositiveReal) extends AnyVal
    
    /** Helpers for working with the rectangular prism in which a spot is analysed */
    object BoxSizeZ:
        /** Sort box size by underlying numerical value. */
        given boxZOrder: Order[BoxSizeZ] = Order.by(_.get)
    end BoxSizeZ

    /** The side length, in 'y', of the bounding region of a spot */
    final case class BoxSizeY(get: PositiveReal) extends AnyVal
    
    /** Helpers for working with the rectangular prism in which a spot is analysed */
    object BoxSizeY:
        /** Sort box size by underlying numerical value. */
        given boxZOrder: Order[BoxSizeY] = Order.by(_.get)
    end BoxSizeY

    /** The side length, in 'x', of the bounding region of a spot */
    final case class BoxSizeX(get: PositiveReal) extends AnyVal
    
    /** Helpers for working with the rectangular prism in which a spot is analysed */
    object BoxSizeX:
        /** Sort box size by underlying numerical value. */
        given boxZOrder: Order[BoxSizeX] = Order.by(_.get)
    end BoxSizeX

end LocusSpotQC
