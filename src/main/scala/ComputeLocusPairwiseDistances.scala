package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.*
import cats.effect.unsafe.implicits.global // for IORuntime
import cats.syntax.all.*
import fs2.data.csv.*
import mouse.boolean.*
import scopt.OParser
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanDistance
import at.ac.oeaw.imba.gerlich.gerlib.imaging.{
    FieldOfViewLike, 
    PositionName,
}
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.FieldOfViewColumnName
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{ ColumnName, NamedRow, readCsvToCaseClasses, writeCaseClassesToCsv }
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.nonnegativeInt.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.* // for .show_ syntax

import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.TraceIdColumnName
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.TraceIdColumnName

/**
 * Simple pairwise distances within trace IDs
 * 
 * @author Vince Reuter
 */
object ComputeLocusPairwiseDistances extends PairwiseDistanceProgram, ScoptCliReaders, StrictLogging:
    /* Constants */
    private val ProgramName = "ComputeLocusPairwiseDistances"
    private val MaxBadRecordsToShow = 3
    
    /** CLI definition */
    final case class CliConfig(
        tracesFile: os.Path = null,
        outputFolder: os.Path = null, 
    )
    val cliParseBuilder = OParser.builder[CliConfig]

    /** Program driver */
    def main(args: Array[String]): Unit = {
        import cliParseBuilder.*

        val parser = OParser.sequence(
            programName(ProgramName),
            // TODO: better naming and versioning
            head(ProgramName, BuildInfo.version),
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
            case Some(opts) => workflow(opts.tracesFile, opts.outputFolder)
        }
    }

    def workflow(inputFile: os.Path, outputFolder: os.Path): Unit = {
        val expOutBaseName = s"${inputFile.last.split("\\.").head}.pairwise_distances__locus_specific"
        val outputFile = HeadedFileWriter.DelimitedTextTarget(outputFolder, expOutBaseName, Delimiter.CommaSeparator).filepath
        val inputDelimiter = Delimiter.fromPathUnsafe(inputFile)
        
        /* Read input, then throw exception or write output. */
        logger.info(s"Reading input file: ${inputFile}")
        val observedOutputFile = Input.parseRecords(inputFile).bimap(_.toNel, inputRecordsToOutputRecords) match {
            case (Some(bads), _) => throw Input.BadRecordsException(bads)
            case (None, outputRecords) => 
                val recs = outputRecords.toList.sortBy{ r => 
                    (r.fieldOfView, r.region, r.trace, r.locus1, r.locus2)
                }(summon[Order[(PositionName, RegionId, TraceId, LocusId, LocusId)]].toOrdering)
                
                logger.info(s"Writing output file: $outputFile")
                if (!os.exists(outputFile.parent)){ os.makeDir.all(outputFile.parent) }
                fs2.Stream.emits(recs)
                    .through(writeCaseClassesToCsv(outputFile))
                    .compile
                    .drain
                    .unsafeRunSync()
                logger.info("Done!")
        }
    }

    def inputRecordsToOutputRecords(inrecs: Iterable[(Input.GoodRecord, NonnegativeInt)]): Iterable[OutputRecord] = {
        inrecs.groupBy((r, _) => Input.getGroupingKey(r)).toList.flatMap{ 
            case ((fov, tid, reg), groupedRecords) => 
                groupedRecords.toList.combinations(2).flatMap{
                    case (r1, i1) :: (r2, i2) :: Nil => (r1.locus =!= r2.locus).option(
                        OutputRecord(
                            fieldOfView = fov,
                            trace = tid,
                            region = reg,
                            locus1 = r1.locus, 
                            locus2 = r2.locus,
                            distance = EuclideanDistance.between(r1.point, r2.point), 
                            inputIndex1 = i1, 
                            inputIndex2 = i2
                            )
                    )
                    case rs => throw new Exception(s"${rs.length} records (not 2) when taking pairs!")
                }
        }
    }

    object Input:
        /* These come from the *traces.csv file produced at the end of looptrace. */
        val FieldOfViewColumn = FieldOfViewColumnName.value
        val TraceIdColumn = TraceIdColumnName.value
        val RegionalBarcodeTimepointColumn = "ref_timepoint"
        val LocusSpecificBarcodeTimepointColumn = "timepoint"
        val XCoordinateColumn = "x"
        val YCoordinateColumn = "y"
        val ZCoordinateColumn = "z"
        
        /**
         * Parse input records from the given file.
         * 
         * @param inputFile The file from which to read records
         * @return A pair in which the first element is a collection of records which failed to parse, 
         *     augmented with line number and with messages about what went wrong during the parse attempt; 
         *     the second element is a collection of pairs of successfully parsed record along with line number
         */
        def parseRecords(inputFile: os.Path): (List[BadInputRecord], List[(GoodRecord, NonnegativeInt)]) = {
            val (header, records) = preparse(inputFile)
            
            def getParser[A](col: String, lift: String => Either[String, A]): ValidatedNel[String, Array[String] => ValidatedNel[String, A]] = 
                getColParser(header)(col, lift)

            /* Component parsers, one for each field of interest from a record. */
            val maybeParseFOV = getParser(FieldOfViewColumn, PositionName.parse)
            val maybeParseTrace = getParser(TraceIdColumn, safeParseInt >>> TraceId.fromInt)
            val maybeParseRegion = getParser(RegionalBarcodeTimepointColumn, safeParseInt >>> RegionId.fromInt)
            val maybeParseLocus = getParser(LocusSpecificBarcodeTimepointColumn, safeParseInt >>> LocusId.fromInt)
            val maybeParseX = getParser(XCoordinateColumn, safeParseDouble >> XCoordinate.apply)
            val maybeParseY = getParser(YCoordinateColumn, safeParseDouble >> YCoordinate.apply)
            val maybeParseZ = getParser(ZCoordinateColumn, safeParseDouble >> ZCoordinate.apply)

            (maybeParseFOV, maybeParseTrace, maybeParseRegion, maybeParseLocus, maybeParseX, maybeParseY, maybeParseZ).mapN(
                (parseFOV, parseTrace, parseRegion, parseLocus, parseX, parseY, parseZ) => 
                    val validateRecordLength = (r: Array[String]) => 
                        (r.size === header.length).either(NonEmptyList.one(s"Header has ${header.length} fields, but line has ${r.size}"), r)
                    Alternative[List].separate(NonnegativeInt.indexed(records).map{ 
                        (r, i) => validateRecordLength(r).flatMap(Function.const{
                            (parseFOV(r), parseTrace(r), parseRegion(r), parseLocus(r), parseX(r), parseY(r), parseZ(r)).mapN(
                                (fov, trace, region, locus, x, y, z) => GoodRecord(fov, trace, region, locus, Point3D(x, y, z))
                            ).toEither
                        }).bimap(msgs => BadInputRecord(i, r.toList, msgs), _ -> i)
                    })
            ).fold(missing => throw IllegalHeaderException(header.toList, missing.toNes), identity)
        }

        /** How records must be grouped for consideration of between which pairs to compute distance */
        def getGroupingKey(r: GoodRecord) = (r.fieldOfView, r.trace, r.region)
        
        /**
         * Wrapper around data representing a successfully parsed record from the input file
         * 
         * @param fieldOfView The field of view (FOV) in which this spot was detected
         * @param trace The identifier of the trace to which this spot belongs
         * @param region The timepoint in which this spot's associated regional barcode was imaged
         * @param time The timepoint in which the (locus-specific) spot was imaged
         * @param point The 3D spatial coordinates of the center of a FISH spot
         */
        final case class GoodRecord(fieldOfView: PositionName, trace: TraceId, region: RegionId, locus: LocusId, point: Point3D)

        /**
         * Bundle of data representing a bad record (line) from input file
         * 
         * @oaram lineNumber The number of the line on which the bad record occurs
         * @param data The raw CSV parse record (key-value mapping)
         * @param errors What went wrong with parsing the record's data
         */
        final case class BadInputRecord(lineNumber: Int, data: List[String], errors: NonEmptyList[String])
        
        /** Helpers for working with bad input records */
        object BadInputRecord:
            given showForBadInputRecord: Show[BadInputRecord] = Show.show{ r => s"${r.lineNumber}: ${r.data} -- ${r.errors}" }
        end BadInputRecord


        /** Error for when at least one record fails to parse correctly. */
        final case class BadRecordsException(records: NonEmptyList[BadInputRecord]) 
            extends Exception(s"${records.length} bad input records; first $MaxBadRecordsToShow (max): ${records.take(MaxBadRecordsToShow)}")
    end Input

    /** Bundler of data which represents a single output record (pairwise distance) */
    final case class OutputRecord(
        fieldOfView: PositionName, 
        trace: TraceId, 
        region: RegionId, 
        locus1: LocusId, 
        locus2: LocusId, 
        distance: EuclideanDistance, 
        inputIndex1: NonnegativeInt, 
        inputIndex2: NonnegativeInt,
    )

    /** Helpers for working with output records */
    object OutputRecord:
        import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given

        given CsvRowEncoder[OutputRecord, String] with
            override def apply(elem: OutputRecord): RowF[Some, String] = 
                val fovText: NamedRow = FieldOfViewColumnName.write(elem.fieldOfView)
                val traceText: NamedRow = TraceIdColumnName.write(elem.trace)
                val regText: NamedRow = ColumnName[RegionId]("region").write(elem.region)
                val loc1Text: NamedRow = ColumnName[LocusId]("locus1").write(elem.locus1)
                val loc2Text: NamedRow = ColumnName[LocusId]("locus2").write(elem.locus2)
                val distanceText: NamedRow = ColumnName[EuclideanDistance]("distance").write(elem.distance)
                val i1Text: NamedRow = ColumnName[NonnegativeInt]("inputIndex1").write(elem.inputIndex1)
                val i2Text: NamedRow = ColumnName[NonnegativeInt]("inputIndex2").write(elem.inputIndex2)
                fovText |+| traceText |+| regText |+| loc1Text |+| loc2Text |+| distanceText |+| i1Text |+| i2Text
    end OutputRecord
end ComputeLocusPairwiseDistances
