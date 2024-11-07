package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.*
import cats.effect.unsafe.implicits.global
import cats.syntax.all.*
import fs2.data.csv.*
import mouse.boolean.*
import scopt.OParser
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanDistance
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.FieldOfViewColumnName
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{ ColumnName, NamedRow, readCsvToCaseClasses, writeCaseClassesToCsv }
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.nonnegativeInt.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.drift.given
import at.ac.oeaw.imba.gerlich.looptrace.drift.DriftRecord
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/**
 * Euclidean distances between pairs of regional barcode spots
 * 
 * @author Vince Reuter
 */
object ComputeRegionPairwiseDistances extends PairwiseDistanceProgram, ScoptCliReaders, StrictLogging:
    /* Constants */
    private val ProgramName = "ComputeRegionPairwiseDistances"
    private val MaxBadRecordsToShow = 3
    
    /** CLI definition */
    final case class CliConfig(
        roisFile: os.Path = null,
        noDriftCorrection: Boolean = false,
        maybeDriftFile: Option[os.Path] = None,
        outputFolder: os.Path = null, 
    )
    val cliParseBuilder = OParser.builder[CliConfig]

    given Eq[os.Path] = Eq.fromUniversalEquals

    /** Program driver */
    def main(args: Array[String]): Unit = {
        import cliParseBuilder.*

        def noDriftCorrectionOptionName = "noDriftCorrection"

        val parser = OParser.sequence(
            programName(ProgramName),
            head(ProgramName, BuildInfo.version),
            opt[os.Path]("roisFile")
                .required()
                .action((f, c) => c.copy(roisFile = f))
                .validate((f: os.Path) => os.isFile(f).either(s"Not an extant file: $f", ())), 
            opt[os.Path]("driftFile")
                .action((f, c) => c.copy(maybeDriftFile = f.some))
                .validate(f => os.isFile(f).either(s"Not an extant file; $f", ()))
                .text("Path to file with per-FOV, per-timepoint shifts to correct for drift"),
            opt[os.Path]('O', "outputFolder")
                .required()
                .action((f, c) =>  c.copy(outputFolder = f))
                .text("Path to output folder in which to write."),
            opt[Unit](noDriftCorrectionOptionName)
                .action((_, c) => c.copy(noDriftCorrection = true))
                .text("Indicate that no drift correction should be done"),
            checkConfig{ c => 
                (c.maybeDriftFile, c.noDriftCorrection) match {
                    case (None, false) => 
                        failure(s"No drift correction file was provided, but --$noDriftCorrectionOptionName was NOT used")
                    case (None, true) => 
                        success
                    case (Some(driftFile), false) => 
                        if driftFile === c.roisFile
                        then failure(s"Drift file is the same as ROIs file: $driftFile")
                        else if (driftFile === c.outputFolder)
                        then failure(s"Drift file is the same as output folder: $driftFile")
                        else success
                    case (Some(f), true) => 
                        failure(s"Drift correction file was provided, but --$noDriftCorrectionOptionName was used")
                }
            },
            checkConfig{ c => 
                if c.roisFile === c.outputFolder
                then failure(s"ROIs file is the same as the output folder: ${c.roisFile}")
                else success
            }
        )
        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => workflow(opts.roisFile, opts.maybeDriftFile, opts.outputFolder)
        }
    }

    def workflow(inputFile: os.Path, maybeDriftFile: Option[os.Path], outputFolder: os.Path): Unit = {        
        /* Read input, then throw exception or write output. */
        logger.info(s"Reading ROIs file: ${inputFile}")
        val (badInputs, goodInputs) = Input.parseRecords(inputFile)
        badInputs.toNel match {
            case Some(bads) => throw new Input.BadRecordsException(bads)
            case None => 
                import fs2.data.text.utf8.byteStreamCharLike
                import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.imaging.given

                val maybeDrifts = maybeDriftFile.map{ driftFile => 
                    logger.info(s"Reading drift file: ${driftFile}")
                    readCsvToCaseClasses[DriftRecord](driftFile).unsafeRunSync()
                }
                val outputRecords = inputRecordsToOutputRecords(goodInputs, maybeDrifts)
                
                val outputFile = HeadedFileWriter.DelimitedTextTarget(
                    outputFolder, 
                    s"${inputFile.last.split("\\.").head}.pairwise_distances__regional",
                    Delimiter.CommaSeparator,
                ).filepath

                logger.info(s"Writing output file: $outputFile")
                fs2.Stream.emits(outputRecords.toList)
                    .through(writeCaseClassesToCsv(outputFile))
                    .compile
                    .drain
                    .unsafeRunSync()
                logger.info("Done!")
        }
    }

    def inputRecordsToOutputRecords(
        inrecs: Iterable[(Input.GoodRecord, NonnegativeInt)], 
        maybeDrifts: Option[List[DriftRecord]],
    ): Iterable[OutputRecord] = 
        val getPoint: Input.GoodRecord => Point3D = maybeDrifts match {
            case None => _.point
            case Some(driftRecords) => 
                import at.ac.oeaw.imba.gerlich.looptrace.drift.Movement
                val keyed = driftRecords.map{ d => (d.fieldOfView, d.time) -> d }.toMap
                (roi: Input.GoodRecord) => 
                    keyed.get(roi.fieldOfView -> roi.region.get) match {
                        case None => throw new Exception(s"No drift for input record $roi")
                        case Some(drift) => Movement.addDrift(drift.total)(roi.point)
                    }
        }
        inrecs.groupBy((r, _) => Input.getGroupingKey(r)).toList.flatMap{ case (fov, groupedRecords) => 
            groupedRecords.toList.combinations(2).map{
                case (r1, i1) :: (r2, i2) :: Nil => 
                    val d = EuclideanDistance.between(getPoint(r1), getPoint(r2))
                    OutputRecord(
                        fieldOfView = fov, 
                        region1 = r1.region, 
                        region2 = r2.region, 
                        distance = d, 
                        inputIndex1 = i1, 
                        inputIndex2 = i2,
                    )
                case rs => throw new Exception(s"${rs.length} records (not 2) when taking pairs!")
            }
        }

    object Input:
        /* These come from the *traces.csv file produced at the end of looptrace. */
        val FieldOfViewColumn = FieldOfViewColumnName.value
        val RegionalBarcodeTimepointColumn = "timepoint"
        val XCoordinateColumn = "xc"
        val YCoordinateColumn = "yc"
        val ZCoordinateColumn = "zc"

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
            val maybeParseRegion = getParser(RegionalBarcodeTimepointColumn, safeParseInt >>> RegionId.fromInt)
            val maybeParseX = getParser(XCoordinateColumn, safeParseDouble >> XCoordinate.apply)
            val maybeParseY = getParser(YCoordinateColumn, safeParseDouble >> YCoordinate.apply)
            val maybeParseZ = getParser(ZCoordinateColumn, safeParseDouble >> ZCoordinate.apply)

            (maybeParseFOV, maybeParseRegion, maybeParseX, maybeParseY, maybeParseZ).mapN(
                (parseFOV, parseRegion, parseX, parseY, parseZ) => 
                    val validateRecordLength = (r: Array[String]) => 
                        (r.size === header.length).either(NonEmptyList.one(s"Header has ${header.length} fields, but line has ${r.size}"), r)
                    Alternative[List].separate(NonnegativeInt.indexed(records).map{ 
                        (r, i) => validateRecordLength(r).flatMap(Function.const{
                            (parseFOV(r), parseRegion(r), parseX(r), parseY(r), parseZ(r)).mapN(
                                (fov, region, x, y, z) => GoodRecord(fov, region, Point3D(x, y, z))
                            ).toEither
                        }).bimap(msgs => BadInputRecord(i, r.toList, msgs), _ -> i)
                    })
            ).fold(missing => throw IllegalHeaderException(header.toList, missing.toNes), identity)
        }

        /** How records must be grouped for consideration of between which pairs to compute distance */
        def getGroupingKey = (_: GoodRecord).fieldOfView
        
        /**
         * Wrapper around data representing a successfully parsed record from the input file
         * 
         * @param fieldOfView The field of view (FOV) in which this spot was detected
         * @param trace The identifier of the trace to which this spot belongs
         * @param region The timepoint in which this spot's associated regional barcode was imaged
         * @param time The timepoint in which the (locus-specific) spot was imaged
         * @param point The 3D spatial coordinates of the center of a FISH spot
         */
        final case class GoodRecord(fieldOfView: PositionName, region: RegionId, point: Point3D)
        
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
        region1: RegionId, 
        region2: RegionId, 
        distance: EuclideanDistance, 
        inputIndex1: NonnegativeInt, 
        inputIndex2: NonnegativeInt
        )

    // Helpers for working with output record instances
    object OutputRecord:
        import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.regionId.given

        private given CellEncoder[EuclideanDistance] = summon[CellEncoder[NonnegativeReal]].contramap(_.get)

        given CsvRowEncoder[OutputRecord, String] with
            override def apply(elem: OutputRecord): RowF[Some, String] = 
                val fovText: NamedRow = FieldOfViewColumnName.write(elem.fieldOfView)
                val r1Text: NamedRow = ColumnName[RegionId]("region1").write(elem.region1)
                val r2Text: NamedRow = ColumnName[RegionId]("region2").write(elem.region2)
                val distanceText: NamedRow = ColumnName[EuclideanDistance]("distance").write(elem.distance)
                val i1Text: NamedRow = ColumnName[NonnegativeInt]("inputIndex1").write(elem.inputIndex1)
                val i2Text: NamedRow = ColumnName[NonnegativeInt]("inputIndex2").write(elem.inputIndex2)
                fovText |+| r1Text |+| r2Text |+| distanceText |+| i1Text |+| i2Text
    end OutputRecord

    /* Type aliases */
    private type DriftKey = (FieldOfViewLike, ImagingTimepoint)
end ComputeRegionPairwiseDistances
