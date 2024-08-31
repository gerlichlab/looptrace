package at.ac.oeaw.imba.gerlich.looptrace
package roi

import cats.data.NonEmptySet

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

/** A roi that's too close to one or more others, such that it's unusable */
final case class UnidentifiableRoi(
    index: RoiIndex, 
    context: ImagingContext,
    centroid: Centroid[Double],
    box: BoundingBox,
    nucleus: NuclearDesignation, 
    partners: NonEmptySet[RoiIndex],
)
