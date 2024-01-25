package at.ac.oeaw.imba.gerlich.looptrace

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
object ComputeLocusPairwiseDistances:
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

    def workflow(inputFile: os.Path, outputFolder: os.Path): Either[String, HeadedFileWriter.DelimitedTextTarget] = {
        val expOutBaseName = s"${inputFile.last.split("\\.").head}.pairwise_distances__locus_specific"
        val expectedOutputFile = HeadedFileWriter.DelimitedTextTarget(outputFolder, expOutBaseName, Delimiter.CommaSeparator)
        val inputDelimiter = Delimiter.fromPathUnsafe(inputFile)
        
        /* Read input, then throw exception or write output. */
        println(s"Reading input file: ${inputFile}")
        val observedOutputFile = Input.parseRecords(inputFile).bimap(_.toNel, inputRecordsToOutputRecords) match {
            case (Some(bads), _) => throw Input.BadRecordsException(bads)
            case (None, outputRecords) => 
                val recs = outputRecords.toList.sortBy{ r => 
                    (r.position, r.region, r.trace, r.locus1, r.locus2)
                }(summon[Order[(PositionIndex, RegionId, TraceId, LocusId, LocusId)]].toOrdering)
                println(s"Writing output file: ${expectedOutputFile.filepath}")
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

    def inputRecordsToOutputRecords(inrecs: Iterable[(Input.GoodRecord, NonnegativeInt)]): Iterable[OutputRecord] = {
        inrecs.groupBy((r, _) => Input.getGroupingKey(r)).toList.flatMap{ 
            case ((pos, tid, reg), groupedRecords) => 
                groupedRecords.toList.combinations(2).flatMap{
                    case (r1, i1) :: (r2, i2) :: Nil => (r1.locus =!= r2.locus).option(
                        OutputRecord(
                            position = pos,
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
        val FieldOfViewColumn = "pos_index"
        val TraceIdColumn = "trace_id"
        val RegionalBarcodeTimepointColumn = "ref_frame"
        val LocusSpecificBarcodeTimepointColumn = "frame"
        val XCoordinateColumn = "x"
        val YCoordinateColumn = "y"
        val ZCoordinateColumn = "z"

        val allColumns = List(
            FieldOfViewColumn, 
            TraceIdColumn, 
            LocusSpecificBarcodeTimepointColumn, 
            RegionalBarcodeTimepointColumn, 
            ZCoordinateColumn,
            YCoordinateColumn, 
            XCoordinateColumn,
            )
        
        /**
         * Parse input records from the given file.
         * 
         * @param inputFile The file from which to read records
         * @return A pair in which the first element is a collection of records which failed to parse, 
         *     augmented with line number and with messages about what went wrong during the parse attempt; 
         *     the second element is a collection of pairs of successfully parsed record along with line number
         */
        def parseRecords(inputFile: os.Path): (List[BadInputRecord], List[(GoodRecord, NonnegativeInt)]) = {
            val (header, records) = os.read.lines(inputFile)
                .map(Delimiter.CommaSeparator.split)
                .toList
                .toNel
                .fold(throw EmptyFileException(inputFile))(recs => recs.head -> recs.tail)
            
            def getParser[A](col: String, lift: String => Either[String, A]): ValidatedNel[String, Array[String] => ValidatedNel[String, A]] = 
                getColParser(header)(col, lift)

            /* Component parsers, one for each field of interest from a record. */
            val maybeParseFOV = getParser(FieldOfViewColumn, safeParseInt >>> PositionIndex.fromInt)
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
        def getGroupingKey(r: GoodRecord) = (r.position, r.trace, r.region)
        
        /**
         * Wrapper around data representing a successfully parsed record from the input file
         * 
         * @param position The field of view (FOV) in which this spot was detected
         * @param trace The identifier of the trace to which this spot belongs
         * @param region The timepoint in which this spot's associated regional barcode was imaged
         * @param time The timepoint in which the (locus-specific) spot was imaged
         * @param point The 3D spatial coordinates of the center of a FISH spot
         */
        final case class GoodRecord(position: PositionIndex, trace: TraceId, region: RegionId, locus: LocusId, point: Point3D)
        
        /** Exception for when necessary columns are missing from header. */
        final case class IllegalHeaderException(header: List[String], missing: NonEmptySet[String]) extends Throwable:
            require(missing.forall(Input.allColumns.contains), s"Alleged missing columns aren't required: ${missing.toList.sorted.mkString(", ")}")
            override def toString = s"header = $header, missing = $missing"

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

        /** Error type for when a file to use is unexpectedly empty. */
        final case class EmptyFileException(getFile: os.Path) extends Exception(s"File is empty: $getFile")

        private def getColParser[A](header: Array[String])(col: String, lift: String => Either[String, A]): ValidatedNel[String, Array[String] => ValidatedNel[String, A]] =
            header.zipWithIndex
                .find(_._1 === col)
                .map((_, i) => safeGetFromRow(i, lift)(_: Array[String]))
                .toRight(col)
                .toValidatedNel
    end Input

    /** Bundler of data which represents a single output record (pairwise distance) */
    final case class OutputRecord(
        position: PositionIndex, 
        trace: TraceId, 
        region: RegionId, 
        locus1: LocusId, 
        locus2: LocusId, 
        distance: EuclideanDistance, 
        inputIndex1: NonnegativeInt, 
        inputIndex2: NonnegativeInt
        )

    /** How to write the output records from this program */
    object OutputWriter extends HeadedFileWriter[OutputRecord]:
        // These are our names.
        override def header: List[String] = List("position", "traceId", "region", "locus1", "locus2", "distance", "inputIndex1", "inputIndex2")
        override def toTextFields(r: OutputRecord): List[String] = r match {
            case OutputRecord(pos, trace, region, locus1, locus2, distance, idx1, idx2) => 
                List(pos.show, trace.show, region.show, locus1.show, locus2.show, distance.get.toString, idx1.show, idx2.show)
        }
    end OutputWriter
end ComputeLocusPairwiseDistances
