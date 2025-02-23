package at.ac.oeaw.imba.gerlich.looptrace
package csv

/** Aggregation of CSV-related typeclass instances */
package object instances:
    object all extends AllCsvInstances

    trait AllCsvInstances extends DriftCsvInstances, GeometryInstances, ImagingCsvInstances, LocusIdInstances, RegionIdInstances, RoiCsvInstances, TracingCsvInstances