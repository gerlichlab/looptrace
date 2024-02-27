package at.ac.oeaw.imba.gerlich.looptrace

import scala.math.max
import scala.util.Try
import upickle.default.*
import cats.{ Alternative, Order }
import cats.data.{ NonEmptyList as NEL, ValidatedNel }
import cats.syntax.apply.*
import cats.syntax.either.*
import cats.syntax.eq.*
import cats.syntax.flatMap.*
import cats.syntax.functor.*
import cats.syntax.option.*
import cats.syntax.order.*
import mouse.boolean.*
import scopt.OParser

import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.LabelAndFilterTracesQC.ParserConfig.traceIdKey
import at.ac.oeaw.imba.gerlich.looptrace.space.{ Point3D, XCoordinate, YCoordinate, ZCoordinate }
import at.ac.oeaw.imba.gerlich.looptrace.syntax.*

/** Label points underlying traces with various QC pass-or-fail values. */
object LabelAndFilterTracesQC:
    val ProgramName = "LabelAndFilterTracesQC"
    val QcPassColumn = "qcPass"
    
    /** Deinition of the command-line interface */
    case class CliConfig(
        configuration: ImagingRoundsConfiguration = null, // unconditionally required
        traces: os.Path = null, // unconditionally required
        maxDistanceToRegionCenter: LocusSpotQC.DistanceToRegion = LocusSpotQC.DistanceToRegion(NonnegativeReal(Double.MaxValue)),
        minSignalToNoise: LocusSpotQC.SignalToNoise = LocusSpotQC.SignalToNoise(PositiveReal(1e-10)),
        maxSigmaXY: LocusSpotQC.SigmaXY = LocusSpotQC.SigmaXY(PositiveReal(Double.MaxValue)),
        maxSigmaZ: LocusSpotQC.SigmaZ = LocusSpotQC.SigmaZ(PositiveReal(Double.MaxValue)),
        probesToIgnore: Seq[ProbeName] = List(),
        minTraceLength: NonnegativeInt = NonnegativeInt(0),
        parserConfig: Option[os.Path] = None, 
        outputFolder: Option[os.Path] = None
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
            fovColumn = "pos_index",
            regionColumn = "ref_frame",
            traceIdColumn = "trace_id",
            timeColumn = "frame",
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

        final case class ParseError(errorMessages: NEL[String]) extends Exception(s"${errorMessages.size} errors: ${errorMessages}")
    end ParserConfig

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import ScoptCliReaders.given
        import parserBuilder.*

        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, VersionName), 
            opt[ImagingRoundsConfiguration]("configuration")
                .required()
                .action((progConf, cliConf) => cliConf.copy(configuration = progConf))
                .text("Path to file specifying the imaging rounds configuration"),
            opt[os.Path]("tracesFile")
                .required()
                .action((f, c) => c.copy(traces = f))
                .validate(f => os.isFile(f).either(s"Alleged traces file isn't a file: $f", ()))
                .text("Path to the traces data file"),
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
            opt[os.Path]('O', "outputFolder")
                .action((d, c) => c.copy(outputFolder = d.some))
                .validate(d => os.isDir(d).either(s"Alleged output folder isn't a directory: $d", ()))
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => 
                val outfolder = opts.outputFolder.getOrElse(opts.traces.parent)
                val parserConfiguration: os.Path | ParserConfig = opts.parserConfig.getOrElse(ParserConfig.default)
                workflow(
                    opts.configuration,
                    parserConfiguration, 
                    opts.traces, 
                    opts.maxDistanceToRegionCenter, 
                    opts.minSignalToNoise, 
                    opts.maxSigmaXY, 
                    opts.maxSigmaZ, 
                    opts.minTraceLength, 
                    outfolder
                    )
        }
    }

    def workflow(
        imagingRoundsConfiguration: ImagingRoundsConfiguration,
        parserConfigPathOrConf: os.Path | ParserConfig, 
        tracesFile: os.Path, 
        maxDistFromRegion: LocusSpotQC.DistanceToRegion, 
        minSignalToNoise: LocusSpotQC.SignalToNoise, 
        maxSigmaXY: LocusSpotQC.SigmaXY, 
        maxSigmaZ: LocusSpotQC.SigmaZ,
        minTraceLength: NonnegativeInt, 
        outfolder: os.Path
        ): Unit = {
        
        val pc: ParserConfig = parserConfigPathOrConf match {
            case c: ParserConfig => c
            case confFile: os.Path => 
                println(s"Reading parser configuration file: $confFile")
                readJsonFile[ParserConfig](confFile)
        }
        
        val delimiter = Delimiter.fromPathUnsafe(tracesFile)
        
        os.read.lines(tracesFile).map(delimiter.split).toList match {
            case (Nil | (_ :: Nil)) => println("Traces file has no records, skipping QC labeling and filtering")
            case header :: records => 
                val maybeParse: ErrMsgsOr[Array[String] => ErrMsgsOr[(TraceSpotId, LocusSpotQC.DataRecord)]] = {
                    val maybeParseFov = buildFieldParse(pc.fovColumn, safeParseInt >>> PositionIndex.fromInt)(header)
                    val maybeParseRegion = buildFieldParse(pc.regionColumn, safeParseInt >>> RegionId.fromInt)(header)
                    val maybeParseTraceId = buildFieldParse(pc.traceIdColumn, safeParseInt >>> TraceId.fromInt)(header)
                    val maybeParseTime = buildFieldParse(pc.timeColumn, safeParseInt >>> Timepoint.fromInt)(header)
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
                    (
                        maybeParseFov,
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
                        maybeParseBoxX
                    ).mapN((
                        parseFov,
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
                        parseBoxX
                        ) => { (record: Array[String]) => 
                            (record.length === header.length).either(NEL.one(s"Record has ${record.length} fields but header has ${header.length}"), ()).flatMap{
                                Function.const{(
                                    parseFov(record),
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
                                    parseBoxX(record)
                                    ).mapN((fov, rid, tid, time, z, y, x, refDist, a, bg, sigXY, sigZ, boxZ, boxY, boxX) => 
                                        val uniqId = TraceSpotId(TraceGroupId(fov, rid, tid), time)
                                        val bounds = LocusSpotQC.BoxUpperBounds(boxX, boxY, boxZ)
                                        val center = Point3D(x, y, z)
                                        val qcData = LocusSpotQC.DataRecord(bounds, center, refDist, a, bg, sigXY, sigZ)
                                        uniqId -> qcData
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
                            (uniqId, qcData: LocusSpotQC.DataRecord) => 
                                val qcResult = qcData.toQCResult(maxDistFromRegion, minSignalToNoise, maxSigmaXY, maxSigmaZ)
                                (uniqId, (rec -> qcResult))
                            )
                        }) match {
                            /* Throw an exception if any error occurred, otherwise write 2 results files. */
                            case (Nil, recordsWithQC) => 
                                val recordsToWrite = recordsWithQC
                                    .filterNot((uniqId, _) => imagingRoundsConfiguration.tracingExclusions.contains(uniqId.time))
                                    .map((uniqId, recAndRes) => uniqId.groupId -> recAndRes)
                                writeResults(header, outfolder, tracesFile.baseName, delimiter)(minTraceLength, recordsToWrite)
                            case (badRecords, _) => throw new Exception(s"${badRecords.length} problem(s) reading records: $badRecords")
                        }
                }
        }
    }

    /** Write filtered and unfiltered results files, filtered having just QC pass flag column uniformly 1, unfiltered having causal components. */
    def writeResults(
        header: Array[String], outfolder: os.Path, basename: String, delimiter: Delimiter
        )(minTraceLength: NonnegativeInt, records: Iterable[(TraceGroupId, (Array[String], LocusSpotQC.ResultRecord))]): Unit = {
        require(os.isDir(outfolder), s"Output folder path isn't a directory: $outfolder")
        val (withinRegionCol, snrCol, denseXYCol, denseZCol, inBoundsXCol, inBoundsYCol, inBoundsZCol) = labelsOf[LocusSpotQC.ResultRecord]
        Alternative[List].separate(NonnegativeInt.indexed(records.toList).map { 
            case (rec@(_, (original, qcResult)), recnum) => 
                (header.length === original.length).either(
                    ((s"Header has ${header.length}, original has ${original.length}"), recnum), 
                    rec
                    )
        }) match {
            case (Nil, unfiltered) => // success (no errors) case --> write output files
                val (actualHeader, finaliseOriginal) = header.head match {
                    case "" => (header.tail, (_: Array[String]).tail)
                    case _ => (header, identity(_: Array[String]))
                }
                
                /* Unfiltered output */
                val getQCFlagsText = (qc: LocusSpotQC.ResultRecord) => (qc.components :+ qc.allPass).map(p => if p then "1" else "0")
                val unfilteredOutputFile = outfolder / s"${basename}.unfiltered.${delimiter.ext}" // would need to update ImageHandler.traces_file_qc_unfiltered if changed
                val unfilteredHeader = actualHeader ++ List(withinRegionCol, snrCol, denseXYCol, denseZCol, inBoundsXCol, inBoundsYCol, inBoundsZCol, QcPassColumn)
                val unfilteredRows = unfiltered.map{ case (_, (original, qc)) => finaliseOriginal(original) ++ getQCFlagsText(qc) }
                println(s"Writing unfiltered output: $unfilteredOutputFile")
                writeTextFile(unfilteredOutputFile, unfilteredHeader :: unfilteredRows, delimiter)

                /* Filtered output */
                val filteredOutputFile = outfolder / s"${basename}.filtered.${delimiter.ext}" // would need to update ImageHandler.traces_file_qc_filtered if changed
                val filteredHeader = actualHeader :+ QcPassColumn
                val filteredRows = unfiltered.flatMap{ 
                    case (groupId, (original, qc)) => qc.allPass.option{ (groupId, finaliseOriginal(original) :+ "1") }
                }
                val hist = filteredRows.groupBy(_._1).view.mapValues(_.length).toMap
                val keepKeys = hist.filter(_._2 >= minTraceLength).keySet
                val recordsToWrite = filteredRows
                    .filter((groupId, _) => keepKeys.contains(groupId))
                    .map((_, fields) => fields)
                println(s"Writing filtered output: $filteredOutputFile")
                writeTextFile(filteredOutputFile, filteredHeader :: recordsToWrite, delimiter)
            
            case (bads, _) => throw new Exception(s"${bads.length} problem(s) with writing results: $bads")
        }
    }

    /** Supports for the same trace must share not only the same {@code TraceId}, but also be from the same FOV and region. */
    final case class TraceGroupId(position: PositionIndex, region: RegionId, trace: TraceId)

    /** A single spot belongs to a trace group. Neither the group ID nor probe ID is unique, but together they are. */
    final case class TraceSpotId(groupId: TraceGroupId, time: Timepoint)
    
    final case class BoxSizeColumnX(get: String) extends AnyVal
    final case class BoxSizeColumnY(get: String) extends AnyVal
    final case class BoxSizeColumnZ(get: String) extends AnyVal

    final case class PointColumnX(get: String) extends AnyVal
    final case class PointColumnY(get: String) extends AnyVal
    final case class PointColumnZ(get: String) extends AnyVal
end LabelAndFilterTracesQC
