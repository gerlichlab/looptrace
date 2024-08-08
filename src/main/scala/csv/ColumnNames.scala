package at.ac.oeaw.imba.gerlich.gerlib.looptrace
package csv

import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnName
import at.ac.oeaw.imba.gerlich.looptrace.space.{
    XCoordinate, 
    YCoordinate,
    ZCoordinate, 
}

/** Collection of names of critical columns from which to parse data */
object ColumnNames:
    /** New box corners */
    val NewMinZ = ColumnName[ZCoordinate]("zMin")
    val NewMaxZ = ColumnName[ZCoordinate]("zMax")
    val NewMinY = ColumnName[YCoordinate]("yMin")
    val NewMaxY = ColumnName[YCoordinate]("yMax")
    val NewMinX = ColumnName[XCoordinate]("xMin")
    val NewMaxX = ColumnName[XCoordinate]("xMax")

    /** Old box corners */
    val OldMinZ = ColumnName[ZCoordinate]("z_min")
    val OldMaxZ = ColumnName[ZCoordinate]("z_max")
    val OldMinY = ColumnName[YCoordinate]("y_min")
    val OldMaxY = ColumnName[YCoordinate]("y_max")
    val OldMinX = ColumnName[XCoordinate]("x_min")
    val OldMaxX = ColumnName[XCoordinate]("x_max")
