package at.ac.oeaw.imba.gerlich.looptrace
package instances

import cats.*
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.imagingTimepoint.given

trait LocusIdInstances:
  import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.imagingTimepoint.given
  given SimpleShow[LocusId] =
    summon[SimpleShow[ImagingTimepoint]].contramap(_.get)
