package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import upickle.default.*
import cats.*
import cats.data.{ EitherNel, NonEmptyList, ValidatedNel }
import cats.syntax.all.*
import mouse.boolean.*
import scopt.OParser
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.imaging.{
    ImagingTimepoint, 
    PositionName,
}
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.instances.simpleShow.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.FieldOfViewColumnName
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.* // for .show_ syntax

import at.ac.oeaw.imba.gerlich.looptrace.HeadedFileWriter.DelimitedTextTarget.eqForDelimitedTextTarget
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.space.{ Point3D, XCoordinate, YCoordinate, ZCoordinate }
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.TraceGroupColumnName

/**
  * Label points underlying traces with various QC pass-or-fail values.
  * 
  * Includes computation of the data needed for `napari` to visualise the locus-specific spots, with QC label info
  * 
  * @author Vince Reuter
  * @see [[https://github.com/gerlichlab/looptrace/issues/269 Issue 269]]
  * @see [[https://github.com/gerlichlab/looptrace/issues/268 Issue 268]]
  * @see [[https://github.com/gerlichlab/looptrace/issues/259 Issue 259]]
  */
object LabelAndFilterLocusSpots extends ScoptCliReaders, StrictLogging:
    val ProgramName = "LabelAndFilterLocusSpots"
    val QcPassColumn = "qcPass"
    
    private type CenterInPixels = Point3D
    private type TraceRecordPair = (NonnegativeInt, LocusSpotQC.OutputRecord)

    /** Deinition of the command-line interface */
    case class CliConfig(
        configuration: ImagingRoundsConfiguration = null, // unconditionally required
        traces: os.Path = null, // unconditionally required
        pointsDataOutputFolder: os.Path = null, // unconditionally required
        roiSizeZ: LocusSpotQC.PixelCountZ = LocusSpotQC.PixelCountZ(PositiveInt(1)), // unconditionally required
        roiSizeY: LocusSpotQC.PixelCountY = LocusSpotQC.PixelCountY(PositiveInt(1)), // unconditionally required
        roiSizeX: LocusSpotQC.PixelCountX = LocusSpotQC.PixelCountX(PositiveInt(1)), // unconditionally required
        maxDistanceToRegionCenter: LocusSpotQC.DistanceToRegion = LocusSpotQC.DistanceToRegion(NonnegativeReal(Double.MaxValue)),
        minSignalToNoise: LocusSpotQC.SignalToNoise = LocusSpotQC.SignalToNoise(PositiveReal(1e-10)),
        maxSigmaXY: LocusSpotQC.SigmaXY = LocusSpotQC.SigmaXY(PositiveReal(Double.MaxValue)),
        maxSigmaZ: LocusSpotQC.SigmaZ = LocusSpotQC.SigmaZ(PositiveReal(Double.MaxValue)),
        probesToIgnore: Seq[ProbeName] = List(),
        minTraceLength: NonnegativeInt = NonnegativeInt(0),
        parserConfig: Option[os.Path] = None, 
        analysisOutputFolder: Option[os.Path] = None, 
        overwrite: Boolean = false,
    )

    /** The definition of how to build the parser for the data (traces) file */
    case class ParserConfig(
        fovColumn: String,
        regionColumn: String,
        traceIdColumn: String,
        timeColumn: String,
        xySigmaColumn: String, 
        zSigmaColumn: String, 
        zPointColumn: PointColumnZ,
        yPointColumn: PointColumnY, 
        xPointColumn: PointColumnX,
        zBoxSizeColumn: BoxSizeColumnZ, 
        yBoxSizeColumn: BoxSizeColumnY, 
        xBoxSizeColumn: BoxSizeColumnX, 
        signalColumn: String,
        backgroundColumn: String, 
        distanceToReferenceColumn: String
        )
    
    /** Helpers for working with the parser configuration */
    object ParserConfig:
        val (fovKey, regionKey, traceIdKey, timeKey, xySigKey, zSigKey, zPtKey, yPtKey, xPtKey, zBoxKey, yBoxKey, xBoxKey, aKey, bgKey, refDistKey) = 
            labelsOf[ParserConfig]
        
        /** Enable reading from and writing to JSON representation. */
        given jsonCodec: ReadWriter[ParserConfig] = readwriter[ujson.Value].bimap(
            pc => ujson.Obj(
                fovKey -> ujson.Str(pc.fovColumn),
                regionKey -> ujson.Str(pc.regionColumn),
                traceIdKey -> ujson.Str(pc.traceIdColumn),
                timeKey -> ujson.Str(pc.timeColumn),
                xySigKey -> ujson.Str(pc.xySigmaColumn),
                zSigKey -> ujson.Str(pc.zSigmaColumn), 
                zPtKey -> ujson.Str(pc.zPointColumn.get), 
                yPtKey -> ujson.Str(pc.yPointColumn.get),
                xPtKey -> ujson.Str(pc.xPointColumn.get),
                zBoxKey -> ujson.Str(pc.zBoxSizeColumn.get),
                yBoxKey -> ujson.Str(pc.yBoxSizeColumn.get), 
                xBoxKey -> ujson.Str(pc.xBoxSizeColumn.get), 
                aKey -> ujson.Str(pc.signalColumn), 
                bgKey -> ujson.Str(pc.backgroundColumn), 
                refDistKey -> ujson.Str(pc.distanceToReferenceColumn)
            ),
            json => {
                val fovNel = safeExtractStr(fovKey)(json)
                val regionNel = safeExtractStr(regionKey)(json)
                val traceIdNel = safeExtractStr(traceIdKey)(json)
                val probeNameNel = safeExtractStr(timeKey)(json)
                val xySigNel = safeExtractStr(xySigKey)(json)
                val zSigNel = safeExtractStr(zSigKey)(json)
                val zPtNel = safeExtract(zPtKey, PointColumnZ.apply)(json)
                val yPtNel = safeExtract(yPtKey, PointColumnY.apply)(json)
                val xPtNel = safeExtract(xPtKey, PointColumnX.apply)(json)
                val zBoxNel = safeExtract(zBoxKey, BoxSizeColumnZ.apply)(json)
                val yBoxNel = safeExtract(yBoxKey, BoxSizeColumnY.apply)(json)
                val xBoxNel = safeExtract(xBoxKey, BoxSizeColumnX.apply)(json)
                val signalNel = safeExtractStr(aKey)(json)
                val bgNel = safeExtractStr(bgKey)(json)
                val refDistNel = safeExtractStr(refDistKey)(json)
                (fovNel, regionNel, traceIdNel, probeNameNel, xySigNel, zSigNel, zPtNel, yPtNel, xPtNel, zBoxNel, yBoxNel, xBoxNel, signalNel, bgNel, refDistNel)
                    .mapN(ParserConfig.apply)
                    .fold(errs => throw new ParseError(errs), identity)
            }
        )

        /** The default definition of how to parse the data relevant here from the traces file */
        def default = ParserConfig(
            fovColumn = FieldOfViewColumnName.value,
            regionColumn = "ref_timepoint",
            traceIdColumn = "traceId",
            timeColumn = "timepoint",
            xySigmaColumn = "sigma_xy", 
            zSigmaColumn = "sigma_z", 
            zPointColumn = PointColumnZ("z"),
            yPointColumn = PointColumnY("y"),
            xPointColumn = PointColumnX("x"),
            zBoxSizeColumn = BoxSizeColumnZ("spot_box_z"), 
            yBoxSizeColumn = BoxSizeColumnY("spot_box_y"), 
            xBoxSizeColumn = BoxSizeColumnX("spot_box_x"), 
            signalColumn = "A", 
            backgroundColumn = "BG", 
            distanceToReferenceColumn = "ref_dist"
        )

        final case class ParseError(errorMessages: NonEmptyList[String]) extends Exception(s"${errorMessages.size} errors: ${errorMessages}")
    end ParserConfig

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import parserBuilder.*

        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, BuildInfo.version), 
            opt[ImagingRoundsConfiguration]("configuration")
                .required()
                .action((progConf, cliConf) => cliConf.copy(configuration = progConf))
                .text("Path to file specifying the imaging rounds configuration"),
            opt[os.Path]("tracesFile")
                .required()
                .action((f, c) => c.copy(traces = f))
                .validate(f => os.isFile(f).either(s"Alleged traces file isn't a file: $f", ()))
                .text("Path to the traces data file"),
            opt[os.Path]("pointsDataOutputFolder")
                .required()
                .action((d, c) => c.copy(pointsDataOutputFolder = d))
                .text("Path to folder in which to place the data to support overlaying spot image visualisation with centroid and QC results"),
            opt[PositiveInt]("roiPixelsZ")
                .required()
                .action((z, c) => c.copy(roiSizeZ = LocusSpotQC.PixelCountZ(z)))
                .text("Number of pixels of each ROI, in Z dimension"),
            opt[PositiveInt]("roiPixelsY")
                .required()
                .action((y, c) => c.copy(roiSizeY = LocusSpotQC.PixelCountY(y)))
                .text("Number of pixels of each ROI, in Y dimension"),
            opt[PositiveInt]("roiPixelsX")
                .required()
                .action((x, c) => c.copy(roiSizeX = LocusSpotQC.PixelCountX(x)))
                .text("Number of pixels of each ROI, in X dimension"),
            opt[NonnegativeReal]("maxDistanceToRegionCenter")
                .required()
                .action((d, c) => c.copy(maxDistanceToRegionCenter = LocusSpotQC.DistanceToRegion(d)))
                .text("Maximum allowed distance between a sigle FISH probe centroid and a regional centroid"),
            opt[PositiveReal]("minSNR")
                .required()
                .action((r, c) => c.copy(minSignalToNoise = LocusSpotQC.SignalToNoise(r)))
                .text("Maximum allowed distance between a sigle FISH probe centroid and a regional centroid"),
            opt[PositiveReal]("maxSigmaXY")
                .required()
                .action((r, c) => c.copy(maxSigmaXY = LocusSpotQC.SigmaXY(r)))
                .text("Maximum allowed standard deviation of Gaussian fit in xy for a record to still be used to support traces"),
            opt[PositiveReal]("maxSigmaZ")
                .required()
                .action((r, c) => c.copy(maxSigmaZ = LocusSpotQC.SigmaZ(r)))
                .text("Maximum allowed standard deviation of Gaussian fit in z for a record to still be used to support traces"),
            opt[os.Path]("parserConfig")
                .action((f, c) => c.copy(parserConfig = f.some))
                .validate(f => os.isFile(f).either(s"Alleged parser config isn't a file: $f", ()))
                .text("Path to file defining how to build the parser"),
            opt[os.Path]("analysisOutputFolder")
                .action((d, c) => c.copy(analysisOutputFolder = d.some))
                .validate(d => os.isDir(d).either(s"Alleged output folder isn't a directory: $d", ()))
                .text("Path to the folder in which to place the filtered and unfiltered CSV files with output records"),
            opt[Unit]("overwrite")
                .action((_, c) => c.copy(overwrite = true))
                .text("Allow overwrite of existing output, otherwise crash")
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => 
                val analysisOutfolder = opts.analysisOutputFolder.getOrElse(opts.traces.parent)
                val parserConfiguration: os.Path | ParserConfig = opts.parserConfig.getOrElse(ParserConfig.default)
                val roiSizeZ = 
                workflow(
                    roiSize = LocusSpotQC.RoiImageSize(opts.roiSizeZ, opts.roiSizeY, opts.roiSizeX),
                    imagingRoundsConfiguration = opts.configuration,
                    parserConfigPathOrConf = parserConfiguration, 
                    tracesFile = opts.traces, 
                    maxDistFromRegion = opts.maxDistanceToRegionCenter, 
                    minSignalToNoise = opts.minSignalToNoise, 
                    maxSigmaXY = opts.maxSigmaXY, 
                    maxSigmaZ = opts.maxSigmaZ, 
                    minTraceLength = opts.minTraceLength, 
                    analysisOutfolder = analysisOutfolder, 
                    pointsOutfolder = opts.pointsDataOutputFolder,
                    overwrite = opts.overwrite,
                    )
        }
    }

    def workflow(
        roiSize: LocusSpotQC.RoiImageSize,
        imagingRoundsConfiguration: ImagingRoundsConfiguration,
        parserConfigPathOrConf: os.Path | ParserConfig, 
        tracesFile: os.Path, 
        maxDistFromRegion: LocusSpotQC.DistanceToRegion, 
        minSignalToNoise: LocusSpotQC.SignalToNoise, 
        maxSigmaXY: LocusSpotQC.SigmaXY, 
        maxSigmaZ: LocusSpotQC.SigmaZ,
        minTraceLength: NonnegativeInt, 
        analysisOutfolder: os.Path, 
        pointsOutfolder: os.Path,
        overwrite: Boolean = false,
        ): Unit = {
        
        val pc: ParserConfig = parserConfigPathOrConf match {
            case c: ParserConfig => c
            case confFile: os.Path => 
                logger.info(s"Reading parser configuration file: $confFile")
                readJsonFile[ParserConfig](confFile)
        }
        
        val delimiter = Delimiter.fromPathUnsafe(tracesFile)
        
        os.read.lines(tracesFile).map(delimiter.split).toList match {
            case (Nil | (_ :: Nil)) => logger.info("Traces file has no records, skipping QC labeling and filtering")
            case header :: records => 
                val maybeParse: ErrMsgsOr[Array[String] => ErrMsgsOr[(LocusSpotQC.SpotIdentifier, LocusSpotQC.InputRecord)]] = {
                    val maybeParseFov = buildFieldParse(pc.fovColumn, PositionName.parse)(header)
                    val maybeParseTraceGroup = buildFieldParse(TraceGroupColumnName.value, TraceGroupOptional.fromString)(header)
                    val maybeParseRegion = buildFieldParse(pc.regionColumn, safeParseInt >>> RegionId.fromInt)(header)
                    val maybeParseTraceId = buildFieldParse(pc.traceIdColumn, safeParseInt >>> TraceId.fromInt)(header)
                    val maybeParseTime = buildFieldParse(pc.timeColumn, safeParseInt >>> ImagingTimepoint.fromInt)(header)
                    val maybeParseZ = buildFieldParse(pc.zPointColumn.get, safeParseDouble >> ZCoordinate.apply)(header)
                    val maybeParseY = buildFieldParse(pc.yPointColumn.get, safeParseDouble >> YCoordinate.apply)(header)
                    val maybeParseX = buildFieldParse(pc.xPointColumn.get, safeParseDouble >> XCoordinate.apply)(header)
                    val maybeParseRefDist = buildFieldParse(
                        pc.distanceToReferenceColumn, 
                        safeParseDouble.andThen(_.flatMap(NonnegativeReal.either).map(LocusSpotQC.DistanceToRegion.apply)),
                        )(header)
                    val maybeParseSignal = buildFieldParse(pc.signalColumn, safeParseDouble >> LocusSpotQC.Signal.apply)(header)
                    val maybeParseBackground = buildFieldParse(pc.backgroundColumn, safeParseDouble >> LocusSpotQC.Background.apply)(header)
                    val maybeParseSigmaXY = buildFieldParse(pc.xySigmaColumn, safeParseDouble)(header)
                    val maybeParseSigmaZ = buildFieldParse(pc.zSigmaColumn, safeParseDouble)(header)
                    val maybeParseBoxZ = buildFieldParse(pc.zBoxSizeColumn.get, safeParsePosNum >> LocusSpotQC.BoxBoundZ.apply)(header)
                    val maybeParseBoxY = buildFieldParse(pc.yBoxSizeColumn.get, safeParsePosNum >> LocusSpotQC.BoxBoundY.apply)(header)
                    val maybeParseBoxX = buildFieldParse(pc.xBoxSizeColumn.get, safeParsePosNum >> LocusSpotQC.BoxBoundX.apply)(header)
                    val maybeParsePixelZ = buildFieldParse("z_px", safeParseDouble >> ZCoordinate.apply)(header)
                    val maybeParsePixelY = buildFieldParse("y_px", safeParseDouble >> YCoordinate.apply)(header)
                    val maybeParsePixelX = buildFieldParse("x_px", safeParseDouble >> XCoordinate.apply)(header)
                    (
                        maybeParseFov,
                        maybeParseTraceGroup,
                        maybeParseRegion, 
                        maybeParseTraceId, 
                        maybeParseTime, 
                        maybeParseZ, 
                        maybeParseY, 
                        maybeParseX, 
                        maybeParseRefDist, 
                        maybeParseSignal, 
                        maybeParseBackground, 
                        maybeParseSigmaXY, 
                        maybeParseSigmaZ, 
                        maybeParseBoxZ, 
                        maybeParseBoxY, 
                        maybeParseBoxX, 
                        maybeParsePixelZ,
                        maybeParsePixelY,
                        maybeParsePixelX,
                    ).mapN((
                        parseFov,
                        parseTraceGroup,
                        parseRegion, 
                        parseTraceId,
                        parseTime, 
                        parseZ, 
                        parseY, 
                        parseX, 
                        parseRefDist, 
                        parseSignal, 
                        parseBackground, 
                        parseSigmaXY, 
                        parseSigmaZ, 
                        parseBoxZ, 
                        parseBoxY, 
                        parseBoxX, 
                        parsePixelZ, 
                        parsePixelY, 
                        parsePixelX,
                        ) => { (record: Array[String]) => 
                            (record.length === header.length).either(NonEmptyList.one(s"Record has ${record.length} fields but header has ${header.length}"), ()).flatMap{
                                Function.const{(
                                    parseFov(record),
                                    parseTraceGroup(record),
                                    parseRegion(record), 
                                    parseTraceId(record),
                                    parseTime(record),
                                    parseZ(record),
                                    parseY(record),
                                    parseX(record),
                                    parseRefDist(record), 
                                    parseSignal(record), 
                                    parseBackground(record), 
                                    parseSigmaXY(record), 
                                    parseSigmaZ(record), 
                                    parseBoxZ(record), 
                                    parseBoxY(record), 
                                    parseBoxX(record),
                                    parsePixelZ(record), 
                                    parsePixelY(record), 
                                    parsePixelX(record), 
                                    ).mapN((fov, traceGroup, rid, tid, time, z, y, x, refDist, a, bg, sigXY, sigZ, boxZ, boxY, boxX, zPix, yPix, xPix) => 
                                        val uniqId = LocusSpotQC.SpotIdentifier(fov, traceGroup, rid, tid, LocusId(time))
                                        val bounds = LocusSpotQC.BoxUpperBounds(boxX, boxY, boxZ)
                                        val physicalCenter = Point3D(x, y, z)
                                        val pixelCenter = Point3D(xPix, yPix, zPix)
                                        val qcData = LocusSpotQC.InputRecord(roiSize, pixelCenter, bounds, physicalCenter, refDist, a, bg, sigXY, sigZ)
                                        (uniqId, qcData)
                                    ).toEither
                                }
                            }
                        }
                    ).toEither
                }
                
                /* Throw an exception if parse construction failed, otherwise use the parser. */
                maybeParse match {
                    case Left(errors) => throw new Exception(s"${errors.length} errors building parser: $errors")
                    case Right(parse) => 
                        Alternative[List].separate(NonnegativeInt.indexed(records).map{ (rec, idx) => parse(rec).bimap(
                            idx -> _, 
                            (uniqId, qcData) => 
                                val qcResult = qcData.toQCResult(maxDistFromRegion, minSignalToNoise, maxSigmaXY, maxSigmaZ)
                                val totalResult = LocusSpotQC.OutputRecord(uniqId, qcData.centerInImageUnits, qcResult)
                                // NB: We've retained the whole original record, so the QC info and napari points info are supplemental.
                                (totalResult, rec)
                            )
                        }) match {
                            /* Throw an exception if any error occurred, otherwise write 2 results files. */
                            case (Nil, recordsWithQC) => 
                                writeResults(imagingRoundsConfiguration)(
                                    recordsWithQC, 
                                    imagingRoundsConfiguration.tracingExclusions, 
                                    minTraceLength, 
                                    header, 
                                    analysisOutfolder, 
                                    pointsOutfolder,
                                    tracesFile.baseName, 
                                    delimiter, 
                                    overwrite = overwrite,
                                    )
                            case (badRecords, _) => throw new Exception(s"${badRecords.length} problem(s) reading records: $badRecords")
                        }
                }
        }
    }

    /** Write filtered and unfiltered results files, filtered having just QC pass flag column uniformly 1, unfiltered having causal components. */
    def writeResults(roundsConfig: ImagingRoundsConfiguration)(
        records: Iterable[(LocusSpotQC.OutputRecord, Array[String])], 
        exclusions: Set[ImagingTimepoint], 
        minTraceLength: NonnegativeInt,
        header: Array[String], 
        analysisOutfolder: os.Path, 
        pointsOutfolder: os.Path,
        basename: String, 
        delimiter: Delimiter,
        overwrite: Boolean = false,
    ): Unit = {
        if (!os.isDir(analysisOutfolder)) { os.makeDir.all(analysisOutfolder) }
        if (!os.isDir(pointsOutfolder)) { os.makeDir.all(pointsOutfolder) }
        // Include placeholder for field for label displayability column, which we don't need for CSV writing (only JSON, handled via codec).
        val (withinRegionCol, snrCol, denseXYCol, denseZCol, inBoundsXCol, inBoundsYCol, inBoundsZCol, _) = labelsOf[LocusSpotQC.ResultRecord]
        Alternative[List].separate(NonnegativeInt.indexed(records.toList).map { case (rec@(_, arr), recnum) => 
            (header.length === arr.length).either( ((s"Header has ${header.length}, original has ${arr.length}"), recnum), rec )
        }) match {
            case (Nil, unfiltered) => // success (no errors) case --> write output files
                val (actualHeader, finaliseOriginal) = header.head match {
                    // Handle the fact that original input may've had index column and therefore an empty first header field.
                    // TODO: https://github.com/gerlichlab/looptrace/issues/261
                    case "" => 
                        logger.warn("First field of CSV header is empty/unnamed!")
                        (header.tail, (_: Array[String]).tail)
                    case _ => (header, identity(_: Array[String]))
                }
                
                val qcPassRepr: String = "1"
                val qcFailRepr: String = "0"

                /* Unfiltered output */
                val getQCFlagsText = (qc: LocusSpotQC.ResultRecord) => (qc.components :+ qc.allPass).map(p => if p then qcPassRepr else qcFailRepr)
                val unfilteredOutputFile = analysisOutfolder / s"${basename}.unfiltered.${delimiter.ext}" // would need to update ImageHandler.traces_file_qc_unfiltered if changed
                val unfilteredHeader = actualHeader ++ List(withinRegionCol, snrCol, denseXYCol, denseZCol, inBoundsXCol, inBoundsYCol, inBoundsZCol, QcPassColumn)
                // Here, still write a record even if its timepoint is in exclusions, as it may be useful to know when such "spots" actually pass QC.
                val unfilteredRows = unfiltered.map{ (outrec, original) => finaliseOriginal(original) ++ getQCFlagsText(outrec.qcResult) }
                logger.info(s"Writing unfiltered output: $unfilteredOutputFile")
                writeTextFile(unfilteredOutputFile, unfilteredHeader :: unfilteredRows, delimiter, overwrite = overwrite)
                
                /* Filtered output */
                val filteredOutputFile = analysisOutfolder / s"${basename}.filtered.${delimiter.ext}" // would need to update ImageHandler.traces_file_qc_filtered if changed
                val filteredHeader = actualHeader :+ QcPassColumn
                val filteredRows = unfiltered.flatMap{ (outrec, original) => 
                    if outrec.qcResult.allPass && !exclusions.contains(outrec.identifier.locusId.get) // For filtered output, use the timepoint exclusion filter in addition to QC.
                    then (outrec.identifier, finaliseOriginal(original) :+ qcPassRepr).some
                    else None
                }
                
                val recordsToWrite = {
                     // Among the records with a timepoint NOT in tracingExclusions, determine which ones are in a group of sufficiently large size.
                    val keepKeys = filteredRows
                        .map(_._1)
                        .groupBy(getGroupId)
                        .view.mapValues(_.length)
                        .toMap
                        .filter(_._2 >= minTraceLength)
                        .keySet
                    // Do the filtration for sufficiently large group size, and finalise the array-like of text fields to write.
                    filteredRows
                        .filter{ (spotId, _) => keepKeys.contains(getGroupId(spotId)) }
                        .map((_, fields) => fields)
                }
                logger.info(s"Writing filtered output: $filteredOutputFile")
                writeTextFile(filteredOutputFile, filteredHeader :: recordsToWrite, delimiter, overwrite = overwrite)
            
                /** Points CSVs for visualisation with `napari` */
                val groupedAndTagged: List[(PositionName, List[(List[(LocusSpotQC.OutputRecord, PointDisplayType)], NonnegativeInt)])] = 
                    unfiltered.map(_._1)
                        .groupBy(_.identifier.fieldOfView)
                        .view.mapValues(_.fproduct(PointDisplayType.forRecord))
                        .toList
                        .sortBy(_._1)(using Order[PositionName].toOrdering)
                        .map{ (pos, posGroup) => 
                            // Within each field of view, order by trace ID, and reset to start counting 0, 1, 2, ...
                            val processedGroup = posGroup.groupBy(_._1.traceId)
                                .toList
                                .sortBy(_._1)(Order[TraceId].toOrdering)
                                .map(_._2)
                            pos -> NonnegativeInt.indexed(processedGroup)
                        }


                val (groupedFailQC, groupedPassQC): (List[(PositionName, List[TraceRecordPair])], List[(PositionName, List[TraceRecordPair])]) = 
                    Alternative[List].separate(groupedAndTagged.flatMap{ (pos, traceGroups) =>
                        val (failed, passed) = Alternative[List].separate(traceGroups.flatMap((g, t) => g.flatMap{
                            case (_, PointDisplayType.Invisible) => None
                            case (r, PointDisplayType.QCFail) => (t, r).asLeft.some
                            case (r, PointDisplayType.QCPass) => (t, r).asRight.some
                        }))
                        List((pos, failed).asLeft, (pos, passed).asRight)
                    })
                
                writePointsForNapari(pointsOutfolder)(groupedFailQC, roundsConfig)
                writePointsForNapari(pointsOutfolder)(groupedPassQC, roundsConfig)
                logger.info("Done!")

            case (bads, _) => throw new Exception(s"${bads.length} problem(s) with writing results: $bads")
        }
    }

    // Ensure that the .zarr suffix which is sometimes present in the 
    private[looptrace] def stripZarrPrefixFromPositionName(rawPosName: String): EitherNel[String, String] = 
        val modified = rawPosName.replaceAll(".zarr", "") // Account for the possibility of the .zarr polluting the true position name.
        val expPrefix = "P"
        val hasExpPrefix = 
            modified.startsWith(expPrefix).validatedNel(s"Missing expected prefix ($expPrefix)", ())
        val hasExpLength = 
            val expLength = expPrefix.length + 4 // Expect exactly 4 digits.
            (modified.length === expLength).validatedNel(s"Unexpected length: ${modified.length}, not $expLength", ())
        val allDigitsAfterPrefix = 
            modified.tail
                .filterNot(_.isDigit)
                .toList.toNel
                .toLeft(())
                .leftMap{ nonDigits => NonEmptyList.one(s"${nonDigits.length} non-digit character after prefix") }
                .toValidated
        (hasExpPrefix, hasExpLength, allDigitsAfterPrefix)
            .tupled
            .map(_ => modified)
            .toEither


    private def writePointsForNapari(folder: os.Path)(groupedByPos: List[(PositionName, List[TraceRecordPair])], roundsConfig: ImagingRoundsConfiguration) = {
        import NapariSortKey.given
        import NapariSortKey.*

        val getOutfileAndHeader: (PositionName, PointDisplayType) => EitherNel[String, (os.Path, List[String])] = 
            (pos: PositionName, qcType: PointDisplayType) => {
                stripZarrPrefixFromPositionName(pos.show_).map{ posNameBase => 
                    val fp = folder /  s"$posNameBase.${qcType.toString.toLowerCase}.csv"
                    val baseHeader = List("regionTime", "traceId", "locusTime", "traceIndex", "timeIndex", "z", "y", "x")
                    val header = qcType match {
                        case PointDisplayType.QCPass => baseHeader
                        case PointDisplayType.QCFail => baseHeader :+ "failCode"
                        case PointDisplayType.Invisible => throw new RuntimeException("Tried to create output file for invisible point type!")
                    }
                    fp -> header
                }
            }
        
        groupedByPos
            .flatMap{ (pos, traceRecordPairs) => traceRecordPairs.toNel.map(pos -> _) }
            .map{ (pos, traceRecordPairs) => 
                val (qcType, addFailCodes): (PointDisplayType, (List[String], List[LocusSpotQC.FailureReason]) => List[String]) = {
                    if (traceRecordPairs.head._2.passesQC) { // These are QC PASS records.
                        val updateFields = (fields: List[String], codes: List[LocusSpotQC.FailureReason]) => 
                            if codes.nonEmpty 
                            then throw new IllegalArgumentException(s"Nonempty fail codes for allegedly QC-passed record (FOV $pos)! $fields")
                            else fields
                        (PointDisplayType.QCPass, updateFields)
                    } else { // These are QC FAIL records.
                        val updateFields = (fields: List[String], codes: List[LocusSpotQC.FailureReason]) => 
                            if codes.isEmpty 
                            then throw new IllegalArgumentException(s"Empty fail codes for allegedly QC-failed record (FOV $pos)! $fields")
                            else fields :+ codes.map(_.abbreviation).mkString(" ")
                        (PointDisplayType.QCFail, updateFields)
                    }
                }
                val (outfile, header) = getOutfileAndHeader(pos, qcType)
                    .leftMap{ problems => 
                        new Exception(
                            s"${problems.length} problem(s) getting header and output file for position name $pos: ${problems.mkString_("; ")}"
                        )
                    }
                    .fold(throw _, identity)
                val outrecs = traceRecordPairs.map{ (t, r) => 
                    val p = r.centerInPixels                    
                    val timeIndex = (
                        for {
                            locTimes <- roundsConfig.lookupReindexedImagingTimepoint
                                .get(r.regionTime)
                                .toRight(s"Missing region time ${r.regionTime} in locusGrouping!")
                            ti <- locTimes
                                .get(r.locusTime)
                                .toRight(s"Missing locus time ${r.locusTime} in locus times for region time ${r.regionTime}!")
                        } yield ti
                    ).fold(msg => throw new Exception(msg), identity)
                    val base = List(r.regionTime.show_, r.traceId.show_, r.locusTime.show_, t.show_, timeIndex.show, p.z.show_, p.y.show_, p.x.show_)
                    addFailCodes(base, r.failureReasons)
                }
                logger.debug(s"Writing locus points visualisation file: $outfile")
                os.write(outfile, (header :: outrecs).map(_.mkString(",") ++ "\n").toList)
                (pos, outfile)
            }
    }

    /** Wrapper around {@code os.write} to handle writing an iterable of lines. */
    private def writeTextFile(target: os.Path, data: Iterable[Array[String]], delimiter: Delimiter, overwrite: Boolean = false) = {
        val lines = data.map(delimiter.join(_: Array[String]) ++ "\n")
        if overwrite then os.write.over(target, lines) else os.write(target, lines)
    }

    /** From a spot identifier, obtain the elements needed to group it by logical tracing unit. */
    def getGroupId(identifier: LocusSpotQC.SpotIdentifier): (PositionName, RegionId, TraceId) = (identifier.fieldOfView, identifier.regionId, identifier.traceId)

    /**
     * A type that can be ordered for visualisation with `napari`, ordering along one or more slider bars for the viewer
     * 
     * @tparam A The type to be ordered
     */
    trait NapariSortable[A]:
        def getSortKey: A => NapariSortKey
    
    object NapariSortable:
        given contravariantForNapariSortable: Contravariant[NapariSortable] with
            def contramap[A, B](s: NapariSortable[A])(f: B => A): NapariSortable[B] = 
                new NapariSortable[B]:
                    def getSortKey: B => NapariSortKey = f `andThen` s.getSortKey

        given napariSortableForSpotIdentifier: NapariSortable[LocusSpotQC.SpotIdentifier] with
            def getSortKey = (ident: LocusSpotQC.SpotIdentifier) => NapariSortKey(ident.fieldOfView, ident.traceId, ident.locusId.get)
        
        given napariSortableForOutputRecord(using ev: NapariSortable[LocusSpotQC.SpotIdentifier]): NapariSortable[LocusSpotQC.OutputRecord] = 
            ev.contramap(_.identifier)
    end NapariSortable
    
    /** Status of a point to display in napari, based on QC results and its position within the particular spot image timepoint image */
    enum PointDisplayType:
        /** Point that has fulfilled all QC criteria */
        case QCPass
        /** Point that has failed at least one QC criterion */
        case QCFail
        /** Case where centroid of the Gaussian fit is outside the bounds of the volume's bounding box */
        case Invisible

    /** Helpers for working with the point display type classification */
    object PointDisplayType:
        given eqForPointDisplayType: Eq[PointDisplayType] = Eq.fromUniversalEquals[PointDisplayType]
        
        def forRecord(r: LocusSpotQC.OutputRecord): PointDisplayType = (r.canBeDisplayed, r.passesQC) match {
            case (false, _) => Invisible
            case (_, true) => QCPass
            case (_, false) => QCFail 
        }
    end PointDisplayType

    private[LabelAndFilterLocusSpots] final case class NapariSortKey(fieldOfView: PositionName, traceId: TraceId, time: ImagingTimepoint)
    object NapariSortKey:
        given orderForNapariSortKey: Order[NapariSortKey] = Order.by{ k => (k.fieldOfView, k.traceId, k.time) }
        extension [A](as: List[A])(using ev: NapariSortable[A])
            def sortForNapari: List[A] = as.sortBy(ev.getSortKey)(orderForNapariSortKey.toOrdering)

    final case class BoxSizeColumnX(get: String) extends AnyVal
    final case class BoxSizeColumnY(get: String) extends AnyVal
    final case class BoxSizeColumnZ(get: String) extends AnyVal

    final case class PointColumnX(get: String) extends AnyVal
    final case class PointColumnY(get: String) extends AnyVal
    final case class PointColumnZ(get: String) extends AnyVal
end LabelAndFilterLocusSpots
