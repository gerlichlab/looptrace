package at.ac.oeaw.imba.gerlich.looptrace
package csv
package instances

import fs2.data.csv.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint

trait RegionIdInstances:
    given cellEncoderForRegionId(
        using encTime: CellEncoder[ImagingTimepoint]
    ): CellEncoder[RegionId] = encTime.contramap(_.get)

end RegionIdInstances
