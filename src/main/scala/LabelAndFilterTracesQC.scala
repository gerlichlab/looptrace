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
import at.ac.oeaw.imba.gerlich.looptrace.space.{ Point3D, XCoordinate, YCoordinate, ZCoordinate }
import at.ac.oeaw.imba.gerlich.looptrace.LabelAndFilterTracesQC.ParserConfig.traceIdKey

/** Label points underlying traces with various QC pass-or-fail values. */
object LabelAndFilterTracesQC:
    val ProgramName = "LabelAndFilterTracesQC"
    val QcPassColumn = "qcPass"

    type Real = Double
    
    /** Deinition of the command-line interface */
    case class CliConfig(
        traces: os.Path = null,
        maxDistanceToRegionCenter: DistanceToRegion = DistanceToRegion(NonnegativeReal(Double.MaxValue)),
        minSignalToNoise: SignalToNoise = SignalToNoise(PositiveReal(1e-10)),
        maxSigmaXY: SigmaXY = SigmaXY(PositiveReal(Double.MaxValue)),
        maxSigmaZ: SigmaZ = SigmaZ(PositiveReal(Double.MaxValue)),
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
        frameNameColumn: String,
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
        val (fovKey, regionKey, traceIdKey, frameKey, xySigKey, zSigKey, zPtKey, yPtKey, xPtKey, zBoxKey, yBoxKey, xBoxKey, aKey, bgKey, refDistKey) = 
            labelsOf[ParserConfig]
        
        /** Enable reading from and writing to JSON representation. */
        given jsonCodec: ReadWriter[ParserConfig] = readwriter[ujson.Value].bimap(
            pc => ujson.Obj(
                fovKey -> ujson.Str(pc.fovColumn),
                regionKey -> ujson.Str(pc.regionColumn),
                traceIdKey -> ujson.Str(pc.traceIdColumn),
                frameKey -> ujson.Str(pc.frameNameColumn),
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
                val frameNel = safeExtractStr(frameKey)(json)
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
                (fovNel, regionNel, traceIdNel, frameNel, xySigNel, zSigNel, zPtNel, yPtNel, xPtNel, zBoxNel, yBoxNel, xBoxNel, signalNel, bgNel, refDistNel)
                    .mapN(ParserConfig.apply)
                    .fold(errs => throw new ParseError(errs), identity)
            }
        )

        /** The default definition of how to parse the data relevant here from the traces file */
        def default = ParserConfig(
            fovColumn = "pos_index",
            regionColumn = "ref_frame",
            traceIdColumn = "trace_id",
            frameNameColumn = "frame_name",
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
            opt[os.Path]("tracesFile")
                .required()
                .action((f, c) => c.copy(traces = f))
                .validate(f => os.isFile(f).either(s"Alleged traces file isn't a file: $f", ()))
                .text("Path to the traces data file"),
            opt[NonnegativeReal]("maxDistanceToRegionCenter")
                .required()
                .action((d, c) => c.copy(maxDistanceToRegionCenter = DistanceToRegion(d)))
                .text("Maximum allowed distance between a sigle FISH probe centroid and a regional centroid"),
            opt[PositiveReal]("minSNR")
                .required()
                .action((r, c) => c.copy(minSignalToNoise = SignalToNoise(r)))
                .text("Maximum allowed distance between a sigle FISH probe centroid and a regional centroid"),
            opt[PositiveReal]("maxSigmaXY")
                .required()
                .action((r, c) => c.copy(maxSigmaXY = SigmaXY(r)))
                .text("Maximum allowed standard deviation of Gaussian fit in xy for a record to still be used to support traces"),
            opt[PositiveReal]("maxSigmaZ")
                .required()
                .action((r, c) => c.copy(maxSigmaZ = SigmaZ(r)))
                .text("Maximum allowed standard deviation of Gaussian fit in z for a record to still be used to support traces"),
            opt[Seq[String]]("exclusions")
                .action((xs, c) => c.copy(probesToIgnore = xs.map(ProbeName.apply)))
                .text("Names of probes to exclude from traces"),
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
                val conf: os.Path | ParserConfig = opts.parserConfig.getOrElse(ParserConfig.default)
                workflow(
                    conf, 
                    opts.traces, 
                    opts.maxDistanceToRegionCenter, 
                    opts.minSignalToNoise, 
                    opts.maxSigmaXY, 
                    opts.maxSigmaZ, 
                    opts.probesToIgnore, 
                    opts.minTraceLength, 
                    outfolder
                    )
        }
    }

    def workflow(
        pathOrConf: os.Path | ParserConfig, 
        tracesFile: os.Path, 
        maxDistFromRegion: DistanceToRegion, 
        minSignalToNoise: SignalToNoise, 
        maxSigmaXY: SigmaXY, 
        maxSigmaZ: SigmaZ,
        probeExclusions: Iterable[ProbeName], 
        minTraceLength: NonnegativeInt, 
        outfolder: os.Path
        ): Unit = {
        
        val conf: ParserConfig = pathOrConf match {
            case pc: ParserConfig => pc
            case confFile: os.Path => 
                println(s"Reading parser configuration file: $confFile")
                readJsonFile[ParserConfig](confFile)
        }
        
        val delimiter = Delimiter.fromPathUnsafe(tracesFile)
        
        os.read.lines(tracesFile).map(delimiter.split).toList match {
            case (Nil | (_ :: Nil)) => println("Traces file has no records, skipping QC labeling and filtering")
            case header :: records => 
                val maybeParse: ErrMsgsOr[Array[String] => ErrMsgsOr[(TraceSpotId, QCData)]] = {
                    val maybeParseFov = buildFieldParse(conf.fovColumn, safeParseInt.fmap(_ >>= PositionIndex.fromInt))(header)
                    val maybeParseRegion = buildFieldParse(conf.regionColumn, safeParseInt.fmap(_ >>= RegionId.fromInt))(header)
                    val maybeParseTraceId = buildFieldParse(conf.traceIdColumn, safeParseInt.fmap(_ >>= TraceId.fromInt))(header)
                    val maybeParseFrame = buildFieldParse(conf.frameNameColumn, _.asRight.map(ProbeName.apply))(header)
                    val maybeParseZ = buildFieldParse(conf.zPointColumn.get, safeParseDouble.andThen(_.map(ZCoordinate.apply)))(header)
                    val maybeParseY = buildFieldParse(conf.yPointColumn.get, safeParseDouble.andThen(_.map(YCoordinate.apply)))(header)
                    val maybeParseX = buildFieldParse(conf.xPointColumn.get, safeParseDouble.andThen(_.map(XCoordinate.apply)))(header)
                    val maybeParseRefDist = buildFieldParse(conf.distanceToReferenceColumn, safeParseDouble.andThen(_.flatMap(NonnegativeReal.either).map(DistanceToRegion.apply)))(header)
                    val maybeParseSignal = buildFieldParse(conf.signalColumn, safeParseDouble.andThen(_.map(Signal.apply)))(header)
                    val maybeParseBackground = buildFieldParse(conf.backgroundColumn, safeParseDouble.andThen(_.map(Background.apply)))(header)
                    val maybeParseSigmaXY = buildFieldParse(conf.xySigmaColumn, safeParseDouble)(header)
                    val maybeParseSigmaZ = buildFieldParse(conf.zSigmaColumn, safeParseDouble)(header)
                    val maybeParseBoxZ = buildFieldParse(conf.zBoxSizeColumn.get, safeParsePosNum.fmap(_.map(BoxSizeZ.apply)))(header)
                    val maybeParseBoxY = buildFieldParse(conf.yBoxSizeColumn.get, safeParsePosNum.fmap(_.map(BoxSizeY.apply)))(header)
                    val maybeParseBoxX = buildFieldParse(conf.xBoxSizeColumn.get, safeParsePosNum.fmap(_.map(BoxSizeX.apply)))(header)
                    (
                        maybeParseFov,
                        maybeParseRegion, 
                        maybeParseTraceId, 
                        maybeParseFrame, 
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
                        parseFrame, 
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
                                    parseFrame(record),
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
                                    ).mapN((fov, rid, tid, frame, z, y, x, refDist, a, bg, sigXY, sigZ, boxZ, boxY, boxX) => 
                                        val uniqId = TraceSpotId(TraceGroupId(fov, rid, tid), frame)
                                        val qcData = QCData((boxZ, boxY, boxX), Point3D(x, y, z), refDist, a, bg, sigXY, sigZ)
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
                            (uniqId, qcData: QCData) => 
                                val (boxZ, boxY, boxX): (BoxSizeZ, BoxSizeY, BoxSizeX) = qcData.box
                                val (z, y, x): (ZCoordinate, YCoordinate, XCoordinate) = qcData.centroid match { case Point3D(x, y, z) => (z, y, x) }
                                val passDist = qcData.distanceToRegion < maxDistFromRegion
                                val passSNR = qcData.passesSNR(minSignalToNoise)
                                val passSigmaXY = 0 < qcData.sigmaXY && qcData.sigmaXY < maxSigmaXY.get
                                val passSigmaZ = 0 < qcData.sigmaZ && qcData.sigmaZ < maxSigmaZ.get
                                val passBoxZ = max(0, qcData.sigmaZ) < z.get && z.get <  boxZ.get - qcData.sigmaZ
                                val passBoxY = max(0, qcData.sigmaXY) < y.get && y.get <  boxY.get - qcData.sigmaXY
                                val passBoxX = max(0, qcData.sigmaXY) < x.get && x.get <  boxX.get - qcData.sigmaXY
                                val qcResult = QCResult(
                                    withinRegion = passDist, 
                                    sufficientSNR = passSNR, 
                                    denseXY = passSigmaXY, 
                                    denseZ = passSigmaZ, 
                                    inBoundsX = passBoxX, 
                                    inBoundsY = passBoxY, 
                                    inBoundsZ = passBoxZ
                                    )
                                (uniqId, (rec -> qcResult))
                            )
                        }) match {
                            /* Throw an exception if any error occurred, otherwise write 2 results files. */
                            case (Nil, recordsWithQC) => 
                                val recordsToWrite = recordsWithQC
                                    .filterNot((uniqId, _) => probeExclusions.toSet.contains(uniqId.probe))
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
        )(minTraceLength: NonnegativeInt, records: Iterable[(TraceGroupId, (Array[String], QCResult))]): Unit = {
        require(os.isDir(outfolder), s"Output folder path isn't a directory: $outfolder")
        val (withinRegionCol, snrCol, denseXYCol, denseZCol, inBoundsXCol, inBoundsYCol, inBoundsZCol) = labelsOf[QCResult]
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
                val getQCFlagsText = (qc: QCResult) => (qc.components :+ qc.all).map(p => if p then "1" else "0")
                val unfilteredOutputFile = outfolder / s"${basename}.unfiltered.${delimiter.ext}"
                val unfilteredHeader = actualHeader ++ List(withinRegionCol, snrCol, denseXYCol, denseZCol, inBoundsXCol, inBoundsYCol, inBoundsZCol, QcPassColumn)
                val unfilteredRows = unfiltered.map{ case (_, (original, qc)) => finaliseOriginal(original) ++ getQCFlagsText(qc) }
                println(s"Writing unfiltered output: $unfilteredOutputFile")
                writeTextFile(unfilteredOutputFile, unfilteredHeader :: unfilteredRows, delimiter)

                /* Filtered output */
                val filteredOutputFile = outfolder / s"${basename}.filtered.${delimiter.ext}"
                val filteredHeader = actualHeader :+ QcPassColumn
                val filteredRows = unfiltered.flatMap{ 
                    case (groupId, (original, qc)) => qc.all.option{ (groupId, finaliseOriginal(original) :+ "1") }
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

    /** A single spot belongs to a trace group. Neither the group ID nor frame ID is unique, but together they are. */
    final case class TraceSpotId(groupId: TraceGroupId, probe: ProbeName)

    /** A bundle of the QC pass/fail components for individual rows/records supporting traces */
    final case class QCResult(withinRegion: Boolean, sufficientSNR: Boolean, denseXY: Boolean, denseZ: Boolean, inBoundsX: Boolean, inBoundsY: Boolean, inBoundsZ: Boolean):
        final def components: Array[Boolean] = Array(withinRegion, sufficientSNR, denseXY, denseZ, inBoundsX, inBoundsY, inBoundsZ)
        final def all: Boolean = components.all
    end QCResult

    final case class DistanceToRegion(get: NonnegativeReal) extends AnyVal
    object DistanceToRegion:
        given distToRegionOrd: Order[DistanceToRegion] = Order.by(_.get)
    end DistanceToRegion

    final case class SignalToNoise(get: PositiveReal) extends AnyVal
    object SignalToNoise:
        given snrOrder: Order[SignalToNoise] = Order.by(_.get)
    end SignalToNoise

    final case class Signal(get: Real) extends AnyVal
    
    final case class Background(get: Real) extends AnyVal

    final case class SigmaXY(get: PositiveReal) extends AnyVal
    object SigmaXY:
        given sigmaXYOrder: Order[SigmaXY] = Order.by(_.get)
    end SigmaXY

    final case class SigmaZ(get: PositiveReal) extends AnyVal
    object SigmaZ:
        given sigmaZOrder: Order[SigmaZ] = Order.by(_.get)
    end SigmaZ

    final case class BoxSizeZ(get: PositiveReal) extends AnyVal
    object BoxSizeZ:
        given boxZOrder: Order[BoxSizeZ] = Order.by(_.get)
    end BoxSizeZ

    final case class BoxSizeY(get: PositiveReal) extends AnyVal
    object BoxSizeY:
        given boxZOrder: Order[BoxSizeY] = Order.by(_.get)
    end BoxSizeY

    final case class BoxSizeX(get: PositiveReal) extends AnyVal
    object BoxSizeX:
        given boxZOrder: Order[BoxSizeX] = Order.by(_.get)
    end BoxSizeX
    
    final case class QCData(
        box: (BoxSizeZ, BoxSizeY, BoxSizeX),
        centroid: Point3D,
        distanceToRegion: DistanceToRegion, 
        signal: Signal, 
        background: Background, 
        sigmaXY: Real, 
        sigmaZ: Real
        ):
        final def x: XCoordinate = centroid.x
        final def y: YCoordinate = centroid.y
        final def z: ZCoordinate = centroid.z
        final def passesSNR(minSNR: SignalToNoise): Boolean = signal.get > minSNR.get * background.get
    end QCData

    final case class BoxSizeColumnX(get: String) extends AnyVal
    final case class BoxSizeColumnY(get: String) extends AnyVal
    final case class BoxSizeColumnZ(get: String) extends AnyVal

    final case class PointColumnX(get: String) extends AnyVal
    final case class PointColumnY(get: String) extends AnyVal
    final case class PointColumnZ(get: String) extends AnyVal
end LabelAndFilterTracesQC
