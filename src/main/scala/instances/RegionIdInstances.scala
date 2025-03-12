package at.ac.oeaw.imba.gerlich.looptrace
package instances

import cats.*
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint

trait RegionIdInstances:
    given (ev: SimpleShow[ImagingTimepoint]) => SimpleShow[RegionId] = ev.contramap(_.get)
