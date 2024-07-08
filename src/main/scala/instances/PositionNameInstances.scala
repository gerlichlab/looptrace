package at.ac.oeaw.imba.gerlich.looptrace
package instances

import cats.*
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow

trait PositionNameInstances:
    given showForPositionName: Show[PositionName] = Show.show(_.get)
    given SimpleShow[PositionName] = SimpleShow.fromShow
end PositionNameInstances
