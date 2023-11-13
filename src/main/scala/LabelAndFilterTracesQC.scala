package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import upickle.default.*
import cats.{ Alternative, Order }
import cats.data.{ NonEmptyList as NEL, ValidatedNel }
import cats.syntax.apply.*
import cats.syntax.either.*
import cats.syntax.eq.*
import cats.syntax.functor.*
import cats.syntax.option.*
import cats.syntax.order.*
import mouse.boolean.*
import scopt.OParser

import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.space.{ Point3D, XCoordinate, YCoordinate, ZCoordinate }

/** Label points underlying traces with various QC pass-or-fail values. */
object LabelAndFilterTracesQC:
    val ProgramName = "LabelAndFilterTracesQC"

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
        val (xySigColKey, zSigColKey, zPtColKey, yPtColKey, xPtColKey, zBoxColKey, yBoxColKey, xBoxColKey, aColKey, bgColKey, refDistColKey) = 
            labelsOf[ParserConfig]
        
        /** Enable reading from and writing to JSON representation. */
        given jsonCodec: ReadWriter[ParserConfig] = readwriter[ujson.Value].bimap(
            pc => ujson.Obj(
                xySigColKey -> ujson.Str(pc.xySigmaColumn),
                zSigColKey -> ujson.Str(pc.zSigmaColumn), 
                zPtColKey -> ujson.Str(pc.zPointColumn.get), 
                yPtColKey -> ujson.Str(pc.yPointColumn.get),
                xPtColKey -> ujson.Str(pc.xPointColumn.get),
                zBoxColKey -> ujson.Str(pc.zBoxSizeColumn.get),
                yBoxColKey -> ujson.Str(pc.yBoxSizeColumn.get), 
                xBoxColKey -> ujson.Str(pc.xBoxSizeColumn.get), 
                aColKey -> ujson.Str(pc.signalColumn), 
                bgColKey -> ujson.Str(pc.backgroundColumn), 
                refDistColKey -> ujson.Str(pc.distanceToReferenceColumn)
            ),
            json => {
                val xySigNel = safeExtract(xySigColKey, identity)(json)
                val zSigNel = safeExtract(zSigColKey, identity)(json)
                val zPtNel = safeExtract(zPtColKey, PointColumnZ.apply)(json)
                val yPtNel = safeExtract(yPtColKey, PointColumnY.apply)(json)
                val xPtNel = safeExtract(xPtColKey, PointColumnX.apply)(json)
                val zBoxNel = safeExtract(zBoxColKey, BoxSizeColumnZ.apply)(json)
                val yBoxNel = safeExtract(yBoxColKey, BoxSizeColumnY.apply)(json)
                val xBoxNel = safeExtract(xBoxColKey, BoxSizeColumnX.apply)(json)
                val signalNel = safeExtract(aColKey, identity)(json)
                val bgNel = safeExtract(bgColKey, identity)(json)
                val refDistNel = safeExtract(refDistColKey, identity)(json)
                (xySigNel, zSigNel, zPtNel, yPtNel, xPtNel, zBoxNel, yBoxNel, xBoxNel, signalNel, bgNel, refDistNel).mapN(ParserConfig.apply).fold(errs => throw new ParseError(errs), identity)
            }
        )

        /** The default definition of how to parse the data relevant here from the traces file */
        def default = ParserConfig(
            xySigmaColumn = "sigma_xy", 
            zSigmaColumn = "sigma_z", 
            zPointColumn = PointColumnZ("z_px"),
            yPointColumn = PointColumnY("y_px"),
            xPointColumn = PointColumnX("x_px"),
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
                opts.parserConfig match {
                    case None => workflow(ParserConfig.default, opts.traces, opts.maxDistanceToRegionCenter, opts.minSignalToNoise, opts.maxSigmaXY, opts.maxSigmaZ, opts.probesToIgnore, opts.minTraceLength, outfolder)
                    case Some(confFile) => workflow(confFile, opts.traces, opts.maxDistanceToRegionCenter, opts.minSignalToNoise, opts.maxSigmaXY, opts.maxSigmaZ, opts.probesToIgnore, opts.minTraceLength, outfolder)
                }
        }
    }
    
    def workflow(
        confFile: os.Path, 
        tracesFile: os.Path, 
        maxDistFromCenter: DistanceToRegion, 
        minSignalToNoise: SignalToNoise, 
        maxSigmaXY: SigmaXY, 
        maxSigmaZ: SigmaZ, 
        probeExclusions: Iterable[ProbeName], 
        minTraceLength: NonnegativeInt, 
        outfolder: os.Path
        ): Unit = {
        println(s"Reading parser configuration file: $confFile")
        val parserConfig = readJsonFile[ParserConfig](confFile)
        workflow(parserConfig, tracesFile, maxDistFromCenter, minSignalToNoise, maxSigmaXY, maxSigmaZ, probeExclusions, minTraceLength, outfolder)
    }

    def workflow(
        conf: ParserConfig, 
        tracesFile: os.Path, 
        maxDistFromRegion: DistanceToRegion, 
        minSignalToNoise: SignalToNoise, 
        maxSigmaXY: SigmaXY, 
        maxSigmaZ: SigmaZ,
        probeExclusions: Iterable[ProbeName], 
        minTraceLength: NonnegativeInt, 
        outfolder: os.Path
        ): Unit = {
        val delimiter = Delimiter.fromPathUnsafe(tracesFile)
        
        os.read.lines(tracesFile).map(delimiter.split).toList match {
            case (Nil | (_ :: Nil)) => println("Traces file has no records, skipping QC labeling and filtering")
            case header :: records => 
                val maybeParse: ErrMsgsOr[Array[String] => ErrMsgsOr[QCData]] = {
                    val maybeParseZ = buildFieldParse(conf.zPointColumn.get, safeParseDouble.andThen(_.map(ZCoordinate.apply)))(header)
                    val maybeParseY = buildFieldParse(conf.yPointColumn.get, safeParseDouble.andThen(_.map(YCoordinate.apply)))(header)
                    val maybeParseX = buildFieldParse(conf.xPointColumn.get, safeParseDouble.andThen(_.map(XCoordinate.apply)))(header)
                    val maybeParseRefDist = buildFieldParse(conf.distanceToReferenceColumn, safeParseDouble.andThen(_.flatMap(NonnegativeReal.either).map(DistanceToRegion.apply)))(header)
                    val maybeParseSignal = buildFieldParse(conf.signalColumn, safeParseDouble.andThen(_.map(Signal.apply)))(header)
                    val maybeParseBackground = buildFieldParse(conf.backgroundColumn, safeParseDouble.andThen(_.map(Background.apply)))(header)
                    val maybeParseSigmaXY = buildFieldParse(conf.xySigmaColumn, safeParseDouble.andThen(_.flatMap(PositiveReal.either).map(SigmaXY.apply)))(header)
                    val maybeParseSigmaZ = buildFieldParse(conf.zSigmaColumn, safeParseDouble.andThen(_.flatMap(PositiveReal.either).map(SigmaZ.apply)))(header)
                    val maybeParseBoxZ = buildFieldParse(conf.zBoxSizeColumn.get, safeParseInt.andThen(_.flatMap(PositiveInt.either).map(BoxSizeZ.apply)))(header)
                    val maybeParseBoxY = buildFieldParse(conf.yBoxSizeColumn.get, safeParseInt.andThen(_.flatMap(PositiveInt.either).map(BoxSizeY.apply)))(header)
                    val maybeParseBoxX = buildFieldParse(conf.xBoxSizeColumn.get, safeParseInt.andThen(_.flatMap(PositiveInt.either).map(BoxSizeX.apply)))(header)
                    (maybeParseZ, maybeParseY, maybeParseX, maybeParseRefDist, maybeParseSignal, maybeParseBackground, maybeParseSigmaXY, maybeParseSigmaZ, maybeParseBoxZ, maybeParseBoxY, maybeParseBoxX).mapN(
                        (parseZ, parseY, parseX, parseRefDist, parseSignal, parseBackground, parseSigmaXY, parseSigmaZ, parseBoxZ, parseBoxY, parseBoxX) => { (record: Array[String]) => 
                            (record.length === header.length).either(NEL.one(s"Record has ${record.length} fields but header has ${header.length}"), ()).flatMap{
                                    Function.const{(
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
                                        ).mapN((z, y, x, refDist, a, bg, sigXY, sigZ, boxZ, boxY, boxX) => 
                                            QCData((boxZ, boxY, boxX), Point3D(x, y, z), refDist, a, bg, sigXY, sigZ)).toEither
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
                            qcData => 
                                val (boxZ, boxY, boxX): (BoxSizeZ, BoxSizeY, BoxSizeX) = qcData.box
                                val (z, y, x): (ZCoordinate, YCoordinate, XCoordinate) = qcData.centroid match { case Point3D(x, y, z) => (z, y, x) }
                                val passDist = qcData.distanceToRegion < maxDistFromRegion
                                val passSNR = qcData.signal.get > minSignalToNoise.get * qcData.background.get
                                val passSigmaXY = qcData.sigmaXY < maxSigmaXY
                                val passSigmaZ = qcData.sigmaZ < maxSigmaZ
                                val passBoxZ = qcData.sigmaZ.get < z.get && z.get <  boxZ.get - qcData.sigmaZ.get
                                val passBoxY = qcData.sigmaXY.get < y.get && y.get <  boxY.get - qcData.sigmaXY.get
                                val passBoxX = qcData.sigmaXY.get < x.get && x.get <  boxX.get - qcData.sigmaXY.get
                                rec -> QCResult(
                                    withinRegion = passDist, 
                                    signalNoiseRatio = passSNR, 
                                    denseXY = passSigmaXY, 
                                    denseZ = passSigmaZ, 
                                    inBoundsX = passBoxX, 
                                    inBoundsY = passBoxY, 
                                    inBoundsZ = passBoxZ
                                    )
                            )
                        }) match {
                            /* Throw an exception if any error occurred, otherwise write 2 results files. */
                            case (Nil, recordsWithQC) => writeResults(header, outfolder, tracesFile.baseName, delimiter)(recordsWithQC)
                            case (badRecords, _) => throw new Exception(s"${badRecords.length} problem(s) reading records: $badRecords")
                        }
                }
        }
    }

    /** Write filtered and unfiltered results files, filtered having just QC pass flag column uniformly 1, unfiltered having causal components. */
    def writeResults(header: Array[String], outfolder: os.Path, basename: String, delimiter: Delimiter)(records: Iterable[(Array[String], QCResult)]): Unit = {
        require(os.isDir(outfolder), s"Output folder path isn't a directory: $outfolder")
        val (withinRegionCol, snrCol, denseXYCol, denseZCol, inBoundsXCol, inBoundsYCol, inBoundsZCol) = labelsOf[QCResult]
        Alternative[List].separate(NonnegativeInt.indexed(records.toList).map { 
            case ((original, qcResult), lineNum) => (header.length =!= original.length).either(original -> lineNum, original -> qcResult)
        }) match {
            case (Nil, unfiltered) => 
                val qcPassCol = "qcPass"
                
                val getQCFlagsText = (qc: QCResult) => (qc.components :+ qc.all).map(p => if p then "1" else "0")
                val unfilteredOutputFile = outfolder / s"${basename}.unfiltered.${delimiter.ext}"
                val unfilteredHeader = header ++ List(withinRegionCol, snrCol, denseXYCol, denseZCol, inBoundsXCol, inBoundsYCol, inBoundsZCol, qcPassCol)
                println(s"Writing unfiltered output: $unfilteredOutputFile")
                writeTextFile(unfilteredOutputFile, unfilteredHeader :: unfiltered.map{ case (original, qc) => original ++ getQCFlagsText(qc) }, delimiter)
                
                val filteredOutputFile = outfolder / s"${basename}.filtered.${delimiter.ext}"
                val filteredHeader = header :+ qcPassCol
                println(s"Writing filtered output: $filteredOutputFile")
                writeTextFile(filteredOutputFile, filteredHeader :: unfiltered.flatMap{ case (original, qc) => qc.all.option(original :+ "1") }, delimiter)
            
            case (bads, _) => throw new Exception(s"${bads.length} problem(s) with writing results: $bads")
        }
    }
        
    /** bundle of the QC pass/fail components for individual rows/records supporting traces */
    final case class QCResult(withinRegion: Boolean, signalNoiseRatio: Boolean, denseXY: Boolean, denseZ: Boolean, inBoundsX: Boolean, inBoundsY: Boolean, inBoundsZ: Boolean):
        final def components: Array[Boolean] = Array(withinRegion, signalNoiseRatio, denseXY, denseZ, inBoundsX, inBoundsY, inBoundsZ)
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

    final case class BoxSizeZ(get: PositiveInt) extends AnyVal
    object BoxSizeZ:
        given boxZOrder: Order[BoxSizeZ] = Order.by(_.get)
    end BoxSizeZ

    final case class BoxSizeY(get: PositiveInt) extends AnyVal
    object BoxSizeY:
        given boxZOrder: Order[BoxSizeY] = Order.by(_.get)
    end BoxSizeY

    final case class BoxSizeX(get: PositiveInt) extends AnyVal
    object BoxSizeX:
        given boxZOrder: Order[BoxSizeX] = Order.by(_.get)
    end BoxSizeX
    
    final case class QCData(
        box: (BoxSizeZ, BoxSizeY, BoxSizeX),
        centroid: Point3D,
        distanceToRegion: DistanceToRegion, 
        signal: Signal, 
        background: Background, 
        sigmaXY: SigmaXY, 
        sigmaZ: SigmaZ
        ):
        def x: XCoordinate = centroid.x
        def y: YCoordinate = centroid.y
        def z: ZCoordinate = centroid.z
    end QCData

    final case class BoxSizeColumnX(get: String) extends AnyVal
    final case class BoxSizeColumnY(get: String) extends AnyVal
    final case class BoxSizeColumnZ(get: String) extends AnyVal

    final case class PointColumnX(get: String) extends AnyVal
    final case class PointColumnY(get: String) extends AnyVal
    final case class PointColumnZ(get: String) extends AnyVal
end LabelAndFilterTracesQC
