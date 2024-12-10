package at.ac.oeaw.imba.gerlich.looptrace
package instances

import cats.*
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt
import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow

trait TraceIdInstances:
    given simpleShowForTraceId(using ev: SimpleShow[NonnegativeInt]): SimpleShow[TraceId] = 
        ev.contramap(_.get)
    
    given simpleShowForTraceGroupId(using ev: SimpleShow[String]): SimpleShow[TraceGroupOptional] = 
        ev.contramap(_.toOption.fold("")(_.get))
end TraceIdInstances
