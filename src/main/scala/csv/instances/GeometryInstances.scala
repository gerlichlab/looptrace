package at.ac.oeaw.imba.gerlich.looptrace
package csv
package instances

import fs2.data.csv.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanDistance
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeReal

/** CSV-related typeclass instances for data types related to geometry */
trait GeometryInstances:
    /** Unwrap the semntically wrapped value and encode it by the simple numeric value. */
    given cellEncoderForEuclideanDistance(
        using enc: CellEncoder[NonnegativeReal]
    ): CellEncoder[EuclideanDistance] = enc.contramap(_.get)
end GeometryInstances
