package at.ac.oeaw.imba.gerlich.looptrace
package csv
package instances

import cats.data.{ NonEmptyList, NonEmptySet }
import cats.syntax.all.*
import mouse.boolean.*
import fs2.data.csv.*
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.NamedRow
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.{
    RoiIndexColumnName,
    TraceGroupColumnName,
    TraceIdColumnName, 
    TracePartnersAreAllPresentColumnName, 
    TracePartnersColumName, 
}
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given

/** CSV-related typeclass instances for trace IDs */
trait TracingCsvInstances:
    given CellDecoderForTraceGroupMaybe: CellDecoder[TraceGroupMaybe] = 
        CellDecoder.instance{ s => 
            TraceGroupMaybe.fromString(s).leftMap{ msg => new DecoderError(msg) }
        }

    given cellDecoderForTraceId(using dec: CellDecoder[NonnegativeInt]): CellDecoder[TraceId] = 
        dec.map(TraceId.apply)

    /** Encode the trace ID by encoding simply the underlying value. */
    given cellEncoderForTraceId(
        using enc: CellEncoder[NonnegativeInt]
    ): CellEncoder[TraceId] = enc.contramap(_.get)

    /** Encoder a (possibly empty) trace group ID by the wrapped value, or empty string. */
    given CellEncoder[TraceGroupMaybe] = 
        import TraceGroupMaybe.toOption
        CellEncoder.instance(_.toOption.fold("")(_.get))

    /** This is useful for when this encoder is being used where the record's ROI ID is 'NOT' being written separately.
     */
    def getCsvRowEncoderForTraceIdAssignmentWithRoiIndex(using 
        CellEncoder[RoiIndex], 
        CellEncoder[TraceId],
        CellEncoder[Boolean],
    ): CsvRowEncoder[TraceIdAssignment, String] = getCsvRowEncoderForTraceIdAssignment(true)

    /** This is useful for when this encoder is being used where the record's ROI ID is being written separately.
     */
    def getCsvRowEncoderForTraceIdAssignmentWithoutRoiIndex(using 
        CellEncoder[RoiIndex], 
        CellEncoder[TraceId],
        CellEncoder[Boolean],
    ): CsvRowEncoder[TraceIdAssignment, String] = getCsvRowEncoderForTraceIdAssignment(false)

    /**
      * Indicate whether or not to include the ROI ID to which the trace assignment applies.
      *
      * @param includeRoiId Whether or not to write out the assignment's ROI ID
      * @param encTraceGroupId How to write out a trace group ID
      * @param encRoiId How to write out a ROI ID
      * @param encTraceId How to write out a trace ID
      * @param encBool How to write out a true/false value
      * @return A CSV encoder for a trace ID assignment, naming each field appropriately
      */
    def getCsvRowEncoderForTraceIdAssignment(includeRoiId: Boolean)(using 
        encRoiId: CellEncoder[RoiIndex],
        encTraceGroupId: CellEncoder[TraceGroupMaybe],
        encTraceId: CellEncoder[TraceId], 
        encBool: CellEncoder[Boolean],
    ): CsvRowEncoder[TraceIdAssignment, String] = new:
        override def apply(elem: TraceIdAssignment): RowF[Some, String] = 
            val maybeRoiIdRow: Option[NamedRow] = 
                includeRoiId.option{ RoiIndexColumnName.write(elem.roiId) }
            val traceIdRow: NamedRow = TraceIdColumnName.write(elem.traceId)
            val (groupIdOpt, partnersOpt, hasAllPartnersOpt) = elem match {
                case ungrouped: TraceIdAssignment.UngroupedRecord => 
                    (Option.empty[TraceGroupId], Option.empty[NonEmptySet[RoiIndex]], Option.empty[Boolean])
                case unmerged: TraceIdAssignment.GroupedAndUnmerged => 
                    (unmerged.groupId.some, None, None)
                case merged: TraceIdAssignment.GroupedAndMerged => 
                    (merged.groupId.some, merged.partners.some, merged.hasAllPartners.some)
            }
            given cellEncoderOptional[A: CellEncoder]: CellEncoder[Option[A]] = new:
                override def apply(cell: Option[A]): String = cell.fold("")(CellEncoder[A].apply)
            val groupIdRow: NamedRow = TraceGroupColumnName.write(TraceGroupMaybe(groupIdOpt))
            val partnersRow: NamedRow = TracePartnersColumName.write(partnersOpt.fold(Set())(_.toSortedSet))
            val hasAllPartnersRow: NamedRow = TracePartnersAreAllPresentColumnName.write(hasAllPartnersOpt)
            val baseFields = NonEmptyList.of(traceIdRow, groupIdRow, partnersRow, hasAllPartnersRow)
            maybeRoiIdRow.fold(baseFields)(_ :: baseFields).reduce
end TracingCsvInstances
