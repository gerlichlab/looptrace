package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import cats.*
import cats.data.*
import cats.syntax.all.*
import mouse.boolean.*
import scopt.OParser

import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.*
import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.*

/**
 * Simple pairwise distances within trace IDs
 * 
 * @author Vince Reuter
 */
object ComputeSimpleDistances {
    val ProgramName = "ComputeSimpleDistances"
    
    case class CliConfig(
        tracesFile: os.Path = null,
        outputFolder: os.Path = null, 
    )
    
    final case class GoodInputRecord(position: PositionIndex, trace: TraceId, region: GroupName, frame: FrameIndex, point: Point3D)
    final case class BadInputRecord(lineNumber: Int, data: CsvRow, errors: NonEmptyList[String])
    object BadInputRecord:
        given showForBadInputRecord: Show[BadInputRecord] = Show.show{ r => s"${r.lineNumber}: ${r.data} -- ${r.errors}" }
    end BadInputRecord

    /** Likely will correspond to regional barcode imaging timepoint */
    final case class GroupName(get: String) extends AnyVal
    object GroupName:
        /** Often, the regional barcode imaging timepoint will be used as the group name, so provide this convenience constructor. */
        def fromFrameIndex(t: FrameIndex): GroupName = GroupName(t.get.show)
        given orderForGroupName: Order[GroupName] = Order.by(_.get)
        given showForGroupName: Show[GroupName] = Show.show(_.get)
    end GroupName
    
    final case class TraceId private(get: NonnegativeInt) extends AnyVal
    object TraceId:
        given orderForTraceId: Order[TraceId] = Order.by(_.get)
        given showForTraceId: Show[TraceId] = Show.show(_.get.toString)
        def fromInt = NonnegativeInt.either >> TraceId.apply
        def fromRoiIndex(i: RoiIndex): TraceId = new TraceId(i.get)
    end TraceId

    final case class OutputRecord(
        position: PositionIndex, 
        trace: TraceId, 
        region: GroupName, 
        frame1: FrameIndex, 
        frame2: FrameIndex, 
        distance: EuclideanDistance, 
        inputIndex1: NonnegativeInt, 
        inputIndex2: NonnegativeInt
        )

    val parserBuilder = OParser.builder[CliConfig]

    // These come from the *traces.csv file produced at the end of looptrace.
    val FieldOfViewColumn = "pos_index"
    val TraceIdColumn = "trace_id"
    val RegionalBarcodeTimepointColumn = "ref_frame"
    val LocusSpecificBarcodeTimepointColun = "frame"
    val XCoordinateColumn = "x"
    val YCoordinateColumn = "y"
    val ZCoordinateColumn = "z"
    val InputColumns = List(
        FieldOfViewColumn, 
        TraceIdColumn, 
        RegionalBarcodeTimepointColumn, 
        LocusSpecificBarcodeTimepointColun, 
        XCoordinateColumn, 
        YCoordinateColumn, 
        ZCoordinateColumn,
        )

    val OutputWriter = new HeadedFileWriter[OutputRecord] {
        // These are our names.
        override def header: List[String] = List("position", "traceId", "region", "frame1", "frame2", "distance", "inputIndex1", "inputIndex2")
        override def toTextFields(r: OutputRecord): List[String] = r match {
            case OutputRecord(pos, trace, region, frame1, frame2, distance, idx1, idx2) => 
                List(pos.show, trace.show, region.show, frame1.show, frame2.show, distance.get.toString, idx1.show, idx2.show)
        }
    }

    /** Exception for when parsed header does not match expected header. */
    final case class UnexpectedHeaderException(observed: List[String], expected: List[String])
        extends Exception(f"Expected ${observed.mkString(", ")} as header but got ${expected.mkString(", ")}"):
        require(observed =!= expected, "Alleged inequality between observed and expected header, but they're equivalent!")

    def parseRecords(inputFile: os.Path): (List[BadInputRecord], List[(GoodInputRecord, NonnegativeInt)]) = {
        val (header, rawRecords) = safeReadAllWithOrderedHeaders(inputFile).fold(throw _, identity)
        if (header =!= InputColumns) throw new UnexpectedHeaderException(header, InputColumns)
        val validateRecordLength = (r: CsvRow) => 
            (r.size === header.length).either(NonEmptyList.one(s"Header has ${header.length} fields, but line has ${r.size}"), r)
        Alternative[List].separate(NonnegativeInt.indexed(rawRecords.toList).map{ (r, i) => 
            validateRecordLength(r).flatMap(_ => 
                val positionNel = safeGetFromRow(FieldOfViewColumn, safeParseInt >>> PositionIndex.fromInt)(r)
                val traceNel = safeGetFromRow(TraceIdColumn, safeParseInt >>> TraceId.fromInt)(r)
                val regionNel = safeGetFromRow(RegionalBarcodeTimepointColumn, GroupName(_).asRight)(r)
                val locusNel = safeGetFromRow(LocusSpecificBarcodeTimepointColun, safeParseInt >>> FrameIndex.fromInt)(r)
                val pointNel = {
                    val xNel = safeGetFromRow(XCoordinateColumn, safeParseDouble >> XCoordinate.apply)(r)
                    val yNel = safeGetFromRow(YCoordinateColumn, safeParseDouble >> YCoordinate.apply)(r)
                    val zNel = safeGetFromRow(ZCoordinateColumn, safeParseDouble >> ZCoordinate.apply)(r)
                    (xNel, yNel, zNel).mapN(Point3D.apply)
                }
                (positionNel, traceNel, regionNel, locusNel, pointNel).mapN(GoodInputRecord.apply).toEither
            ).bimap(BadInputRecord(i, r, _), _ -> i)
        })
    }

    def main(args: Array[String]): Unit = {
        import parserBuilder.*
        import ScoptCliReaders.given

        val parser = OParser.sequence(
            programName(ProgramName),
            // TODO: better naming and versioning
            head(ProgramName, VersionName),
            opt[os.Path]('T', "tracesFile")
                .required()
                .action((f, c) => c.copy(tracesFile = f))
                .validate((f: os.Path) => os.isFile(f).either(s"Not an extant file: $f", ())), 
            opt[os.Path]('O', "outputFolder")
                .required()
                .action((f, c) =>  c.copy(outputFolder = f))
                .text("Path to output folder in which to write."),
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => workflow(opts.tracesFile, opts.outputFolder).fold(msg => throw new Exception(msg), _ => println("Done!"))
        }
    }

    def inputRecordsToOutputRecords(inrecs: Iterable[(GoodInputRecord, NonnegativeInt)]): Iterable[OutputRecord] = {
        inrecs.groupBy((r, _) => r.position -> r.trace).toList.flatMap{ case ((pos, tid), groupedRecords) => 
            groupedRecords.toList.combinations(2).flatMap{
                case (r1, i1) :: (r2, i2) :: Nil => (r1.region === r2.region && r1.frame =!= r2.frame).option(
                    OutputRecord(
                        position = pos,
                        trace = tid,
                        region = r1.region,
                        frame1 = r1.frame, 
                        frame2 = r2.frame,
                        distance = EuclideanDistance.between(r1.point, r2.point), 
                        inputIndex1 = i1, 
                        inputIndex2 = i2
                        )
                )
                case rs => throw new Exception(s"${rs.length} records (not 2) when taking pairs!")
            }
        }
    }

    def workflow(inputFile: os.Path, outputFolder: os.Path): Either[String, HeadedFileWriter.DelimitedTextTarget] = {
        val expOutBaseName = s"${inputFile.last.split("\\.").head}.pairwise_distances"
        val expectedOutputFile = HeadedFileWriter.DelimitedTextTarget(outputFolder, expOutBaseName, Delimiter.CommaSeparator)
        
        val inputDelimiter = Delimiter.fromPathUnsafe(inputFile)
        println(s"Reading input file: ${inputFile}")
        val (badInputRecords, goodInputRecords) = parseRecords(inputFile)
        if (badInputRecords.nonEmpty) {
            val limit = 3
            println(s"WARNING! ${badInputRecords.length} bad record(s) from input file ${inputFile}")
            println(s"First $limit (below):")
            badInputRecords.take(limit).map(_.show).foreach(println)
        }
        val outputRecords = inputRecordsToOutputRecords(goodInputRecords)
        println(s"Writing output file: ${expectedOutputFile.filepath}")
        val observedOutputFile = {
            val recs = outputRecords.toList.sortBy{ r => 
                (r.position, r.region, r.trace, r.frame1, r.frame2)
            }(summon[Order[(PositionIndex, GroupName, TraceId, FrameIndex, FrameIndex)]].toOrdering)
            OutputWriter.writeRecordsToFile(recs, expectedOutputFile)
        }

        // Facilitate derivation of Eq[HeadedFileWriter[DelimitedTextTarget]].
        import HeadedFileWriter.DelimitedTextTarget.given
        given eqForPath: Eq[os.Path] = Eq.by(_.toString)
        // Check that the observed output path matches the expectation and provide new exception if not.
        (observedOutputFile === expectedOutputFile).either(
            s"Observed output filepath (${observedOutputFile}) differs from expectation (${expectedOutputFile})", 
            observedOutputFile
            )
    }

}
