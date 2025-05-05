package at.ac.oeaw.imba.gerlich.looptrace
package csv
package instances

import fs2.data.csv.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint

/** CSV-related typeclass instances for region IDs */
trait RegionIdInstances:
  /** Encode the region ID by encoding simply the underlying imaging timepoint.
    */
  given (encTime: CellEncoder[ImagingTimepoint]) => CellEncoder[RegionId] =
    encTime.contramap(_.get)
end RegionIdInstances
