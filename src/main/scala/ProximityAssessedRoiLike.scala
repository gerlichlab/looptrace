package at.ac.oeaw.imba.gerlich.looptrace

/** Ability to get a ROI identifier and identifiers of a value's neighboring ROIs */
trait ProximityExclusionAssessedRoiLike[-T]:
    def getRoiIndex: T => RoiIndex
    def getTooCloseNeighbors: T => Set[RoiIndex]
end ProximityExclusionAssessedRoiLike

/** Ability to get a ROI identifier and identifiers of a value's neighboring ROIs */
trait ProximityMergeAssessedRoiLike[-T]:
    def getRoiIndex: T => RoiIndex
    def getMergeNeighbors: T => Set[RoiIndex]
end ProximityMergeAssessedRoiLike
