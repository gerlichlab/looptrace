package at.ac.oeaw.imba.gerlich.looptrace

import cats.data.NonEmptySet

/** The trace ID assignment for a ROI designates whether it's a singleton or
  * part of a merger.
  */
enum TraceIdAssignment:
  /** Regardless of subtype, an assignment has indicates the ROI to which it
    * applies.
    */
  def roiId: RoiIndex

  /** Regardless of subtype, an assignment designates an identifier. */
  def traceId: TraceId

  /** A singleton (for purposes of tracing) ROI for which the regional timepoint
    * 'is NOT' in a group
    */
  case UngroupedRecord(roiId: RoiIndex, traceId: TraceId)
      extends TraceIdAssignment

  /** A singleton (for purposes of tracing) ROI for which the regional timepoint
    * 'is' in a group
    */
  case GroupedAndUnmerged(
      roiId: RoiIndex,
      traceId: TraceId,
      groupId: TraceGroupId
  ) extends TraceIdAssignment

  /** A ROI being merged with others for tracing */
  case GroupedAndMerged(
      roiId: RoiIndex,
      traceId: TraceId,
      groupId: TraceGroupId,
      partners: NonEmptySet[RoiIndex],
      hasAllPartners: Boolean
  ) extends TraceIdAssignment
