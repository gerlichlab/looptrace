package at.ac.oeaw.imba.gerlich.looptrace
package csv

import cats.data.NonEmptySet

import at.ac.oeaw.imba.gerlich.gerlib.geometry.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.PositionName
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnName
import at.ac.oeaw.imba.gerlich.looptrace.drift.*

/** Collection of names of critical columns from which to parse data */
object ColumnNames:
    val MergedIndexColumnName: ColumnName[RoiIndex] = ColumnName("mergeIndex")

    val MergeRoisColumnName: ColumnName[Set[RoiIndex]] = ColumnName(mergeContributorsName)

    /** Distinguished from [[MergedRoisColumnName]] by static guarantee of nonemptiness */
    val MergeContributorsColumnName: ColumnName[NonEmptySet[RoiIndex]] = ColumnName(mergeContributorsName)

    val PositionNameColumnName: ColumnName[PositionName] = ColumnName("position")

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

    private def coarseDriftColumnName[A <: EuclideanAxis](a: A): ColumnName[CoarseDriftComponent[A]] = 
        val coarseDriftColumnSuffix: String = "CoarseDriftPixels"
        ColumnName(a match {
            case EuclideanAxis.X => "x" ++ coarseDriftColumnSuffix
            case EuclideanAxis.Y => "y" ++ coarseDriftColumnSuffix
            case EuclideanAxis.Z => "z" ++ coarseDriftColumnSuffix
        })

    def fineDriftColumnName[A <: EuclideanAxis](a: A): ColumnName[FineDriftComponent[A]] = 
        val fineDriftColumnSuffix: String = "FineDriftPixels"
        ColumnName(a match {
            case EuclideanAxis.X => "x" ++ fineDriftColumnSuffix
            case EuclideanAxis.Y => "y" ++ fineDriftColumnSuffix
            case EuclideanAxis.Z => "z" ++ fineDriftColumnSuffix
        })

    private def mergeContributorsName: String = "mergeRois"
end ColumnNames
