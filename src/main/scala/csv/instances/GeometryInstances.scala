package at.ac.oeaw.imba.gerlich.looptrace
package csv
package instances

import fs2.data.csv.*
import at.ac.oeaw.imba.gerlich.gerlib.geometry.{Distance, EuclideanDistance}

/** CSV-related typeclass instances for data types related to geometry */
trait GeometryInstances:
  /** Unwrap the semntically wrapped value and encode it by the simple numeric
    * value.
    */
  given (enc: CellEncoder[Distance]) => CellEncoder[EuclideanDistance] =
    enc.contramap(_.get)
end GeometryInstances
