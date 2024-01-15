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

/** Simple pairwise distances within trace IDs. */
object ComputeSimpleDistances {
    val ProgramName = "ComputeSimpleDistances"
    
    case class CliConfig(
        tracesFile: os.Path = null,
        outputFolder: os.Path = null, 
        handleExtantOutput: ExtantOutputHandler = null,
        sort: Boolean = false
    )
    
    final case class GoodInputRecord(position: PositionIndex, trace: TraceId, region: GroupName, frame: FrameIndex, point: Point3D)
    
    case class BadInputRecord(lineNumber: Int, data: CsvRow, errors: NonEmptyList[String])
    object BadInputRecord:
        given showForBadInputRecord: Show[BadInputRecord] = Show.show{ r => s"${r.lineNumber}: ${r.data} -- ${r.errors}" }
    end BadInputRecord

    final case class GroupName(get: String) extends AnyVal
    object GroupName:
        given orderForGroupName: Order[GroupName] = Order.by(_.get)
        given showForGroupName: Show[GroupName] = Show.show(_.get)
    end GroupName
    
    final case class TraceId private(get: NonnegativeInt) extends AnyVal
    object TraceId:
        given orderForTraceId: Order[TraceId] = Order.by(_.get)
        given showForTraceId: Show[TraceId] = Show.show(_.get.toString)
        def either = NonnegativeInt.either >> TraceId.apply
        def parse = safeParseInt >>> either
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
    val InputColumns = List("pos_index", "trace_id", "ref_frame", "frame", "x", "y", "z")

    val OutputWriter = new HeadedFileWriter[OutputRecord] {
        // These are our names.
        override def header: Array[String] = Array("position", "traceId", "region", "frame1", "frame2", "distance", "inputIndex1", "inputIndex2")
        override def toTextFields(r: OutputRecord): Array[String] = r match {
            case OutputRecord(pos, trace, region, frame1, frame2, distance, idx1, idx2) => 
                Array(pos.show, trace.show, region.show, frame1.show, frame2.show, distance.get.toString, idx1.show, idx2.show)
        }
        override val delimiter: Delimiter = Delimiter.CommaSeparator
    }

    // TODO: generalise beginning with DistanceRecord.scala.
    //def parseRecords(header: Array[String])(rawRecords: Iterable[Array[String]]): (List[BadInputRecord], List[(GoodInputRecord, NonnegativeInt)]) = {
    def parseRecords(inputFile: os.Path): (List[BadInputRecord], List[(GoodInputRecord, NonnegativeInt)]) = {
        val (header, rawRecords) = safeReadAllWithOrderedHeaders(inputFile).fold(throw _, identity)
        val validateRecordLength = (r: CsvRow) => 
            (r.size === header.length).either(NonEmptyList.one(s"Header has ${header.length} fields, but line has ${r.size}"), r)
        Alternative[List].separate(NonnegativeInt.indexed(rawRecords.toList).map{ (r, i) => 
            validateRecordLength(r).flatMap(_ => 
                val positionNel = safeGetFromRow("pos_index", safeParseInt >>> PositionIndex.fromInt)(r)
                val traceNel = safeGetFromRow("trace_id", TraceId.parse)(r)
                val regionNel = safeGetFromRow("ref_frame", GroupName(_).asRight)(r)
                val locusNel = safeGetFromRow("frame", safeParseInt >>> FrameIndex.fromInt)(r)
                val pointNel = {
                    val xNel = safeGetFromRow("x", safeParseDouble >> XCoordinate.apply)(r)
                    val yNel = safeGetFromRow("y", safeParseDouble >> YCoordinate.apply)(r)
                    val zNel = safeGetFromRow("z", safeParseDouble >> ZCoordinate.apply)(r)
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
            opt[ExtantOutputHandler]("handleExtantOutput")
                .required()
                .action((handle, c) => c.copy(handleExtantOutput = handle))
                .text("How to handle existing output"),
            opt[Unit]('S', "sort")
                .action((_, c) => c.copy(sort = true))
                .text("Say that output should be sorted (ascending by (FOV, region, trace, time))")
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => {
                workflow(opts.tracesFile, opts.outputFolder, opts.handleExtantOutput, sort = opts.sort) match {
                    case Right(_) => println("Done!")
                    case Left(msg: String) => println(msg)
                    case Left(err: Throwable) => throw err
                }
            }
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

    def workflow(inputFile: os.Path, outputFolder: os.Path, handleExtantOutput: ExtantOutputHandler): Either[Throwable | String, HeadedFileWriter.DelimitedTextTarget] = 
        workflow(inputFile, outputFolder, handleExtantOutput, false)

    def workflow(inputFile: os.Path, outputFolder: os.Path, handleExtantOutput: ExtantOutputHandler, sort: Boolean): Either[Throwable | String, HeadedFileWriter.DelimitedTextTarget] = {
        import HeadedFileWriter.*
        import HeadedFileWriter.DelimitedTextTarget.*

        /* Facilitate derivation of Eq[HeadedFileWriter[DelimitedTextTarget]]. */
        import DelimitedTextTarget.given
        given eqForPath: Eq[os.Path] = Eq.by(_.toString)

        val strict = false
        val expectedOutputFile = HeadedFileWriter.DelimitedTextTarget(outputFolder, "pairwise_distances", OutputWriter.delimiter)
        
        handleExtantOutput.prepareToWrite(expectedOutputFile.filepath) match {
            case Right(_) => ()
            case Left(msg: String) => { println(msg); sys.exit(0) }
            case Left(err: Throwable) => throw err
        }
        val inputDelimiter = Delimiter.fromPathUnsafe(inputFile)
        println(s"Reading input file: ${inputFile}")
        val (badInputRecords, goodInputRecords) = parseRecords(inputFile)
        if (badInputRecords.nonEmpty) {
            val limit = 3
            val msg1 = s"${badInputRecords.length} bad record(s) from input file ${inputFile}"
            val msg2 = s"First $limit: ${badInputRecords.take(limit).map(_.show) `mkString` "\n"}"
            if (strict) { throw new Exception(s"$msg1. $msg2") }
            else {
                println(s"WARNING! $msg1")
                println(msg2)
            }
        }
        val outputRecords = inputRecordsToOutputRecords(goodInputRecords)
        println(s"Writing output file: ${expectedOutputFile.filepath}")
        val observedOutputFile = {
            val recs = {
                type Key = (PositionIndex, GroupName, TraceId, FrameIndex, FrameIndex)
                if sort 
                then outputRecords.toList.sortBy(r => (r.position, r.region, r.trace, r.frame1, r.frame2))(summon[Order[Key]].toOrdering)
                else outputRecords
            }
            OutputWriter.writeRecordsToFile(recs)(expectedOutputFile)
        }
        (observedOutputFile === expectedOutputFile).either(
            new Exception(s"Observed output file (${observedOutputFile}) differs from expectation (${expectedOutputFile})"), 
            observedOutputFile
            )
    }

}
