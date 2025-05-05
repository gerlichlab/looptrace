package at.ac.oeaw.imba.gerlich.looptrace
package drift

import at.ac.oeaw.imba.gerlich.gerlib.geometry.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*

final case class DriftRecord(
    fieldOfView: FieldOfViewLike,
    time: ImagingTimepoint,
    coarse: CoarseDrift,
    fine: FineDrift
):
  def total =
    // For justification of additivity, see: https://github.com/gerlichlab/looptrace/issues/194
    TotalDrift(
      DriftComponent.total[AxisZ](coarse.z.value + fine.z.value),
      DriftComponent.total[AxisY](coarse.y.value + fine.y.value),
      DriftComponent.total[AxisX](coarse.x.value + fine.x.value)
    )
end DriftRecord
