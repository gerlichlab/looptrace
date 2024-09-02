package at.ac.oeaw.imba.gerlich.looptrace
package csv

import cats.data.NonEmptySet

import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnName

/** Collection of names of critical columns from which to parse data */
object ColumnNames:
    val RoiIndexColumnName: ColumnName[RoiIndex] = ColumnName("index")

    val MergedIndexColumnName: ColumnName[RoiIndex] = ColumnName("mergeIndex")

    val MergeRoisColumnName: ColumnName[Set[RoiIndex]] = ColumnName(mergeContributorsName)

    /** Distinguished from [[MergedRoisColumnName]] by static guarantee of nonemptiness */
    val MergeContributorsColumnName: ColumnName[NonEmptySet[RoiIndex]] = ColumnName(mergeContributorsName)

    private def mergeContributorsName: String = "mergeRois"
end ColumnNames
