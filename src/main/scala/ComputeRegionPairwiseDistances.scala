package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.*
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import cats.syntax.all.*
import fs2.data.csv.*
import fs2.data.text.utf8.byteStreamCharLike
import mouse.boolean.*
import scopt.OParser
import squants.space.{ Length, LengthUnit, Nanometers }
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.geometry.{
    Centroid,
    EuclideanDistance,
}
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.{
    FieldOfViewColumnName, 
    SpotChannelColumnName, 
}
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{ ColumnName, NamedRow, readCsvToCaseClasses, writeCaseClassesToCsv }
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.drift.{ DriftRecord, Movement }
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.space.LengthInNanometers.given
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.csv.getCsvRowDecoderForImagingChannel
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.TraceGroupColumnName

/**
 * Euclidean distances between pairs of regional barcode spots
 * 
 * @author Vince Reuter
 */
object ComputeRegionPairwiseDistances extends ScoptCliReaders, StrictLogging:
    /* Constants */
    private val ProgramName = "ComputeRegionPairwiseDistances"
    private val MaxBadRecordsToShow = 3
    
    /** CLI definition */
    final case class CliConfig(
        roisFile: os.Path = null, // required
        noDriftCorrection: Boolean = false, // required iff no drift file
        maybeDriftFile: Option[os.Path] = None, // required iff no specification of no drift correction
        pixels: Pixels3D = null, // required
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
                .validate(f => os.isFile(f).either(s"Not an extant file: $f", ()))
                .text("Path to file with per-FOV, per-timepoint shifts to correct for drift"),
            opt[Pixels3D]("pixels")
                .required()
                .action((ps, c) => c.copy(pixels = ps))
                .text("How many nanometers per unit in each direction (x, y, z)"),
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
            case Some(opts) => workflow(opts.roisFile, opts.maybeDriftFile, opts.pixels, opts.outputFolder)
        }
    }

    def workflow(inputFile: os.Path, maybeDriftFile: Option[os.Path], pixels: Pixels3D, outputFolder: os.Path): Unit = {
        import Input.GoodRecord.given

        given RowDec[ImagingChannel] = 
            getCsvRowDecoderForImagingChannel(SpotChannelColumnName)

        /* Read input, then throw exception or write output. */
        val readInput: IO[List[Input.GoodRecord]] = for {
            _ <- IO{ logger.info(s"Reading ROIs file: ${inputFile}") }
            records <- readCsvToCaseClasses[Input.GoodRecord](inputFile)
        } yield records

        val readDrifts: IO[Option[List[DriftRecord]]] = 
            maybeDriftFile.traverse(readCsvToCaseClasses[DriftRecord])

        val program: IO[Unit] = for {
            inrecs <- readInput
            maybeDrifts <- readDrifts
            outrecs = inputRecordsToOutputRecords(inrecs, maybeDrifts, pixels)
            outfile = HeadedFileWriter.DelimitedTextTarget(
                outputFolder, 
                s"${inputFile.last.split("\\.").head}.pairwise_distances__regional",
                Delimiter.CommaSeparator,
            ).filepath
            _ <- writeOutput(outfile, outrecs)
        } yield ()
        
        program.unsafeRunSync()
        logger.info("Done!")
    }

    def writeOutput(outputFile: os.Path, outputRecords: List[OutputRecord]): IO[Unit] = for {
        _ <- IO{ logger.info(s"Writing output file: $outputFile") }
        outputFolder = outputFile.parent
        _ <- IO{ if (!os.exists(outputFolder)) os.makeDir.all(outputFolder) }
        _ <- fs2.Stream.emits(outputRecords.toList)
            .through(writeCaseClassesToCsv(outputFile))
            .compile
            .drain
    } yield ()

    def inputRecordsToOutputRecords(
        inrecs: List[Input.GoodRecord], 
        maybeDrifts: Option[List[DriftRecord]],
        OurPixels: Pixels3D,
    ): List[OutputRecord] = 
        val getPoint: Input.GoodRecord => Point3D = 
            maybeDrifts.fold((_: Input.GoodRecord).point){ driftRecords => 
                val keyed = driftRecords.map{ d => (d.fieldOfView, d.time) -> d }.toMap
                (roi: Input.GoodRecord) => 
                    keyed.get(roi.fieldOfView -> roi.timepoint) match {
                        case None => throw new Exception(s"No drift for input record $roi")
                        case Some(drift) => Movement.addDrift(drift.total)(roi.point)
                    }
            }
        val computeDistance: (Point3D, Point3D) => LengthInNanometers = (p1, p2) => 
            // TODO: replace with a reworked version of Euclidean distance.
            val MyUnit = Nanometers
            val xDiff = (OurPixels.liftX(p1.x.value - p2.x.value) in MyUnit).value
            val yDiff = (OurPixels.liftY(p1.y.value - p2.y.value) in MyUnit).value
            val zDiff = (OurPixels.liftZ(p1.z.value - p2.z.value) in MyUnit).value
            val d = MyUnit(scala.math.sqrt(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff))
            LengthInNanometers.unsafeFromSquants(d)
        inrecs.groupBy(Input.getGroupingKey)
            .toList
            .flatMap{ case ((fov, channel), groupedRecords) => 
                groupedRecords.toList.combinations(2).map{
                    case r1 :: r2 :: Nil => 
                        OutputRecord(
                            fieldOfView = fov, 
                            channel = channel,
                            timepoint1 = r1.timepoint, 
                            timepoint2 = r2.timepoint, 
                            distance = computeDistance(getPoint(r1), getPoint(r2)), 
                            roiId1 = r1.index, 
                            roiId2 = r2.index,
                            groupId1 = r1.traceGroup,
                            groupId2 = r2.traceGroup
                        )
                    case rs => throw new Exception(s"${rs.length} records (not 2) when taking pairs!")
                }
            }
            .sortBy{ r => (r.fieldOfView, r.channel, r.timepoint1, r.timepoint2, r.distance) }(using 
                summon[Order[(PositionName, ImagingChannel, ImagingTimepoint, ImagingTimepoint, LengthInNanometers)]].toOrdering
            )

    private type RowDec[A] = CsvRowDecoder[A, String]

    object Input:
        /** How records must be grouped for consideration of between which pairs to compute distance */
        private[looptrace] def getGroupingKey = (r: GoodRecord) => r.fieldOfView -> r.channel
        
        /**
         * Wrapper around data representing a successfully parsed record from the input file
         * 
         * @param index The ID of the spot/ROI corresponding to this record
         * @param fieldOfView The field of view (FOV) in which this spot was detected
         * @param timepoint The timepoint in which this spot was imaged
         * @param channel The image channel in which the spot was detected
         * @param point The 3D spatial coordinates of the center of a FISH spot
         * @param traceGroup Optionally, the trace group/structure in which this ROI participates
         */
        private[looptrace] final case class GoodRecord(
            index: RoiIndex, 
            fieldOfView: PositionName, 
            timepoint: ImagingTimepoint, 
            channel: ImagingChannel, 
            point: Point3D,
            traceGroup: TraceGroupMaybe,
        )

        private[looptrace] object GoodRecord:
            given csvDecoderForGoodRecord(using 
                decId: RowDec[RoiIndex],
                decPos: CellDecoder[PositionName], 
                decTime: RowDec[ImagingTimepoint], 
                decChannel: RowDec[ImagingChannel], 
                decPoint: RowDec[Centroid[Double]],
                decTraceGroup: CellDecoder[TraceGroupMaybe],
            ): RowDec[GoodRecord] = new:
                override def apply(row: RowF[Some, String]): DecoderResult[GoodRecord] = 
                    val idNel = decId(row)
                        .leftMap{ e => s"Cannot decode ROI ID from row ($row): ${e.getMessage}" }
                        .toValidatedNel
                    val posNel = ColumnName[PositionName](FieldOfViewColumnName.value).from(row)
                    val timeNel = decTime(row)
                        .leftMap{ e => s"Cannot decode timepoint from row ($row): ${e.getMessage}" }
                        .toValidatedNel
                    val channelNel = decChannel(row)
                        .leftMap{ e => s"Cannot decode imaging channel from row ($row): ${e.getMessage}" }
                        .toValidatedNel
                    val pointNel = decPoint(row)
                        .map(_.asPoint)
                        .leftMap{ e => s"Cannot decode 3D point from row ($row): ${e.getMessage}" }
                        .toValidatedNel
                    val traceGroupNel = TraceGroupColumnName.from(row)
                    (idNel, posNel, timeNel, channelNel, pointNel, traceGroupNel)
                        .mapN(GoodRecord.apply)
                        .leftMap{ messages => 
                            DecoderError(s"${messages.length} error(s) decoding input record: ${messages.mkString_("; ")}")
                        }
                        .toEither
        end GoodRecord
        
        /**
         * Bundle of data representing a bad record (line) from input file
         * 
         * @param lineNumber The number of the line on which the bad record occurs
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
        channel: ImagingChannel,
        timepoint1: ImagingTimepoint, 
        timepoint2: ImagingTimepoint, 
        distance: LengthInNanometers, 
        roiId1: RoiIndex, 
        roiId2: RoiIndex,
        groupId1: TraceGroupMaybe, 
        groupId2: TraceGroupMaybe,
        )

    // Helpers for working with output record instances
    object OutputRecord:
        import at.ac.oeaw.imba.gerlich.gerlib.io.csv.fromSimpleShow // CellEncoder.fromSimpleShow extension

        given CellEncoder[LengthInNanometers] = CellEncoder.fromSimpleShow[LengthInNanometers]

        given CsvRowEncoder[OutputRecord, String] with
            override def apply(elem: OutputRecord): RowF[Some, String] = 
                val fovText: NamedRow = FieldOfViewColumnName.write(elem.fieldOfView)
                val channelText: NamedRow = SpotChannelColumnName.write(elem.channel)
                val r1Text: NamedRow = ColumnName[ImagingTimepoint]("timepoint1").write(elem.timepoint1)
                val r2Text: NamedRow = ColumnName[ImagingTimepoint]("timepoint2").write(elem.timepoint2)
                val distanceText: NamedRow = 
                    ColumnName[LengthInNanometers]("distance").write(elem.distance)
                val i1Text: NamedRow = ColumnName[RoiIndex]("roiId1").write(elem.roiId1)
                val i2Text: NamedRow = ColumnName[RoiIndex]("roiId2").write(elem.roiId2)
                val g1Text: NamedRow = TraceGroupColumnName.write(elem.groupId1)
                val g2Text: NamedRow = TraceGroupColumnName.write(elem.groupId2)
                fovText |+| channelText |+| r1Text |+| r2Text |+| distanceText |+| i1Text |+| i2Text |+| g1Text |+| g2Text
    end OutputRecord

    /* Type aliases */
    private type DriftKey = (FieldOfViewLike, ImagingTimepoint)
end ComputeRegionPairwiseDistances
