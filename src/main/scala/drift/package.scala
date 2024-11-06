package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import cats.Order
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given

/** Types and functionality related to image drift */
package object drift:
    private[looptrace] sealed trait DriftComponentLike[V]:
        type Axis <: EuclideanAxis
        def value: V
    end DriftComponentLike

    private[looptrace] case class CoarseDriftComponent[A <: EuclideanAxis : [A] =>> NotGiven[A =:= EuclideanAxis]] private[drift](value: Int) extends DriftComponentLike[Int]:
        override type Axis = A

    private[looptrace] case class FineDriftComponent[A <: EuclideanAxis : [A] =>> NotGiven[A =:= EuclideanAxis]] private[drift](value: Double) extends DriftComponentLike[Double]:
        override type Axis = A

    private[looptrace] case class TotalDriftComponent[A <: EuclideanAxis : [A] =>> NotGiven[A =:= EuclideanAxis]] private[drift](value: Double) extends DriftComponentLike[Double]:
        override type Axis = A

    /** Helpers for working with components of shift to correct for drift. */
    object DriftComponent:
        def coarse[A <: EuclideanAxis : [A] =>> NotGiven[A =:= EuclideanAxis]](value: Int): CoarseDriftComponent[A] = 
            new CoarseDriftComponent[A](value)
        
        def fine[A <: EuclideanAxis : [A] =>> NotGiven[A =:= EuclideanAxis]](value: Double): FineDriftComponent[A] = 
            new FineDriftComponent[A](value)
        
        def total[A <: EuclideanAxis : [A] =>> NotGiven[A =:= EuclideanAxis]](value: Double): TotalDriftComponent[A] = 
            new TotalDriftComponent[A](value)
    end DriftComponent

    final case class CoarseDrift(
        z: CoarseDriftComponent[AxisZ], 
        y: CoarseDriftComponent[AxisY], 
        x: CoarseDriftComponent[AxisX],
    )

    final case class FineDrift(
        z: FineDriftComponent[AxisZ], 
        y: FineDriftComponent[AxisY], 
        x: FineDriftComponent[AxisX],
    )

    final case class TotalDrift(
        z: TotalDriftComponent[AxisZ], 
        y: TotalDriftComponent[AxisY], 
        x: TotalDriftComponent[AxisX],
    )

    /** Tools for working with drift-related shifts */
    object Movement:
        // Raw coordinate type, based on total drift
        private type C = Double
        
        def shiftBy(del: TotalDriftComponent[AxisX])(c: XCoordinate[C]): XCoordinate[C] = XCoordinate(del.value) |+| c
        
        def shiftBy(del: TotalDriftComponent[AxisY])(c: YCoordinate[C]): YCoordinate[C] = YCoordinate(del.value) |+| c
        
        def shiftBy(del: TotalDriftComponent[AxisZ])(c: ZCoordinate[C]): ZCoordinate[C] = ZCoordinate(del.value) |+| c
        
        def addDrift(drift: TotalDrift)(pt: Point3D[C]): Point3D[C] = 
            Point3D(shiftBy(drift.x)(pt.x), shiftBy(drift.y)(pt.y), shiftBy(drift.z)(pt.z))
        
        def addDrift(drift: TotalDrift)(box: BoundingBox[C]): BoundingBox[C] = BoundingBox(
            BoundingBox.Interval(shiftBy(drift.x)(box.sideX.lo), shiftBy(drift.x)(box.sideX.hi)),
            BoundingBox.Interval(shiftBy(drift.y)(box.sideY.lo), shiftBy(drift.y)(box.sideY.hi)), 
            BoundingBox.Interval(shiftBy(drift.z)(box.sideZ.lo), shiftBy(drift.z)(box.sideZ.hi)),
        )
    end Movement
end drift
