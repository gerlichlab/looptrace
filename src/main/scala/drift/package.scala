package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import at.ac.oeaw.imba.gerlich.gerlib.geometry.*

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
end drift
