package at.ac.oeaw.imba.gerlich.looptrace
package csv
package instances

import fs2.data.csv.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint

/** CSV-related typeclass instances for locus IDs */
trait LocusIdInstances:
    /** Encode the locus ID by encoding simply the underlying imaging timepoint. */
    given (encTime: CellEncoder[ImagingTimepoint]) => CellEncoder[LocusId] = encTime.contramap(_.get)
end LocusIdInstances
