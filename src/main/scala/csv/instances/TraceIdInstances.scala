package at.ac.oeaw.imba.gerlich.looptrace
package csv
package instances

import fs2.data.csv.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt

/** CSV-related typeclass instances for trace IDs */
trait TraceIdInstances:
    /** Encode the trace ID by encoding simply the underlying value. */
    given cellEncoderForTraceId(
        using enc: CellEncoder[NonnegativeInt]
    ): CellEncoder[TraceId] = enc.contramap(_.get)
end TraceIdInstances
