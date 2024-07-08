package at.ac.oeaw.imba.gerlich.looptrace
package instances

import cats.*
import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow

trait ProbeNameInstances:
    given showForProbeName: Show[ProbeName] = Show.show(_.get)
    given SimpleShow[ProbeName] = SimpleShow.fromShow
end ProbeNameInstances
