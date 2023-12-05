package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import cats.Alternative
import cats.data.{ NonEmptyList as NEL, Validated, ValidatedNel }
import cats.instances.tuple.*
import cats.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.*

/**
  * Represent a single line/record from the simple pairwise distance computation after tracing.
  *
  * @param position The field of view from which the two points were taken
  * @param trace The trace ID from which the two points were taken
  * @param region The frame/probe/timepoint of the regional barcode in which the points occur
  * @param frame1 One of the two locus-specific FISH probes/timepoints/frames in the pair
  * @param frame2 The other of the two locus-specific FISH probes/timepoints/frames in the pair
  * @param distance The computed (Euclidean) distance between the Gaussian fits' centroids
  * @param inputIndex1 Line index (0-based, excluding header) in the filtered, enriched traces file
  * @param inputIndex2 Line index (0-based, excluding header) in the filtered, enriched traces file
  */
final case class SimpleDistancesRecord(
    position: PositionIndex, 
    traceId: TraceId, 
    region: FrameIndex, 
    frame1: FrameIndex, 
    frame2: FrameIndex, 
    distance: NonnegativeReal, // the distance between the respective centroids of a pair of Gaussian fits
    inputIndex1: LineNumber, // corresponds to line number of filtered, enriched traces file
    inputIndex2: LineNumber // corresponds to line number of filtered, enriched traces file
)
end SimpleDistancesRecord

/** Helpers for working with simple pairwise distance records */
object SimpleDistancesRecord:
    def parse1(row: CsvRow): Validated[NEL[String], SimpleDistancesRecord] = {
        val posNel = safeGetFromRow("position", safeParseInt >>> PositionIndex.fromInt)(row)
        val traceNel = safeGetFromRow("traceId", safeParseInt >>> TraceId.fromInt)(row)
        val regNel = safeGetFromRow("region", safeParseInt >>> FrameIndex.fromInt)(row)
        val frame1Nel = safeGetFromRow("frame1", safeParseInt >>> FrameIndex.fromInt)(row)
        val frame2Nel = safeGetFromRow("frame2", safeParseInt >>> FrameIndex.fromInt)(row)
        val distanceNel = safeGetFromRow("distance", safeParseDouble >>> NonnegativeReal.either)(row)
        val idx1Nel = safeGetFromRow("inputIndex1", safeParseInt >>> NonnegativeInt.either)(row)
        val idx2Nel = safeGetFromRow("inputIndex2", safeParseInt >>> NonnegativeInt.either)(row)
        (posNel, traceNel, regNel, frame1Nel, frame2Nel, distanceNel, idx1Nel, idx2Nel).mapN(SimpleDistancesRecord.apply)
    }

    /** Parse a CSV file with the distance records. */
    def readFile(f: os.Path): Either[Throwable | NEL[(LineNumber, NEL[String])], List[SimpleDistancesRecord]] = {
        safeReadAllWithHeaders(f).flatMap{ rows => 
            val (bads, goods) = Alternative[List].separate{
                NonnegativeInt.indexed(rows).map((r, i) => parse1(r).toEither.leftMap(i -> _))
            }
            bads.toNel.toLeft(goods)
        }
    }
end SimpleDistancesRecord
