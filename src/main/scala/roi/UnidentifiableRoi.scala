package at.ac.oeaw.imba.gerlich.looptrace
package roi

import cats.data.NonEmptySet
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

final case class UnidentifiableRoi(
    index: RoiIndex,
    context: ImagingContext,
    centroid: Centroid[Double],
    box: BoundingBox,
    tooClose: NonEmptySet[RoiIndex]
)
