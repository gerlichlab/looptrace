package at.ac.oeaw.imba.gerlich.looptrace
package syntax

import scala.util.Try

trait SyntaxForJson:
    extension (v: ujson.Value)
        def int: Int = tryToInt(v.num).fold(msg => throw new ujson.Value.InvalidData(v, msg), identity)

    extension (v: ujson.Value)
        def safeInt = Try{ v.int }.toEither
