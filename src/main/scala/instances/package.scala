package at.ac.oeaw.imba.gerlich.looptrace

package object instances:
    object all extends AllInstances

    private trait AllInstances extends 
        LocusIdInstances,
        PositionIndexInstances, 
        PositionNameInstances, 
        ProbeNameInstances, 
        RegionIdInstances, 
        RoiIdInstances, 
        TraceIdInstances