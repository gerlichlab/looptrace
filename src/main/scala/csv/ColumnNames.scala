package at.ac.oeaw.imba.gerlich.looptrace
package csv

import cats.data.NonEmptySet

import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2
import at.ac.oeaw.imba.gerlich.gerlib.geometry.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.PositionName
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnName
import at.ac.oeaw.imba.gerlich.looptrace.drift.*
import at.ac.oeaw.imba.gerlich.looptrace.space.RawCoordinate

/** Collection of names of critical columns from which to parse data */
object ColumnNames:
    /** Each merge input is associated with at least one merge output. */
    val MergeOutputColumnName: ColumnName[RoiIndex] = 
        ColumnName("mergeOutput")

    /** The associated collection may be empty (nothing to merge with). */
    val MergeContributorsColumnNameForAssessedRecord: ColumnName[Set[RoiIndex]] = 
        ColumnName(mergeContributorsName)

    /** Distinguished from [[MergedRoisColumnName]] by static guarantee of at least 2 members */
    val MergeContributorsColumnNameForMergedRecord: ColumnName[AtLeast2[Set, RoiIndex]] = 
        ColumnName(mergeContributorsName)

    val RoiIndexColumnName: ColumnName[RoiIndex] = ColumnName("index")

    val CoarseDriftColumnNameX: ColumnName[CoarseDriftComponent[AxisX]] = 
        coarseDriftColumnName[AxisX](EuclideanAxis.X)
    
    val CoarseDriftColumnNameY: ColumnName[CoarseDriftComponent[AxisY]] = 
        coarseDriftColumnName[AxisY](EuclideanAxis.Y)

    val CoarseDriftColumnNameZ: ColumnName[CoarseDriftComponent[AxisZ]] = 
        coarseDriftColumnName[AxisZ](EuclideanAxis.Z)

    val FineDriftColumnNameX: ColumnName[FineDriftComponent[AxisX]] = 
        fineDriftColumnName[AxisX](EuclideanAxis.X)
    
    val FineDriftColumnNameY: ColumnName[FineDriftComponent[AxisY]] = 
        fineDriftColumnName[AxisY](EuclideanAxis.Y)

    val FineDriftColumnNameZ: ColumnName[FineDriftComponent[AxisZ]] = 
        fineDriftColumnName[AxisZ](EuclideanAxis.Z)

    val TooCloseRoisColumnName: ColumnName[NonEmptySet[RoiIndex]] = 
        ColumnName("tooCloseRois")

    val TraceGroupColumnName: ColumnName[TraceGroupMaybe] = ColumnName("traceGroup")

    val TraceIdColumnName: ColumnName[TraceId] = ColumnName("traceId")

    val TracePartnersAreAllPresentColumnName: ColumnName[Option[Boolean]] = ColumnName("hasAllTracePartners")
    
    val TracePartnersColumName: ColumnName[Set[RoiIndex]] = ColumnName("tracePartners")

    private def coarseDriftColumnName[A <: EuclideanAxis](a: A): ColumnName[CoarseDriftComponent[A]] = 
        ColumnName(a.toString.toLowerCase ++ "DriftCoarsePixels")

    private def fineDriftColumnName[A <: EuclideanAxis](a: A): ColumnName[FineDriftComponent[A]] = 
        ColumnName(a.toString.toLowerCase ++ "DriftFinePixels")

    private def mergeContributorsName: String = "mergePartners"
end ColumnNames
