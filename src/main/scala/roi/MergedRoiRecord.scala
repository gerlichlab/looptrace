package at.ac.oeaw.imba.gerlich.looptrace

import cats.data.*

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox

/** A record of an ROI after the merge process has been considered and done. */
final case class MergedRoiRecord(
    index: RoiIndex, 
    context: ImagingContext, // must be identical among all merge partners
    centroid: Centroid[Double], // averaged over merged partners
    box: BoundingBox, 
    nucleus: NuclearDesignation,  // must be identical among all merge partners
    partners: NonEmptySet[RoiIndex], 
)
