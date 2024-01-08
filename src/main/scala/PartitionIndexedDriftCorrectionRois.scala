package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ Random, Try }
import cats.Alternative
import cats.data.{ NonEmptyList as NEL, ValidatedNel }
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*
import scopt.OParser

import space.CoordinateSequence
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.*

/** Split pool of detected bead ROIs into those for drift correction shift, drift correction accuracy, and unused. */
object PartitionIndexedDriftCorrectionRois:
    val ProgramName = "PartitionIndexedDriftCorrectionRois"

    val BeadRoisPrefix = "bead_rois_"

    /* Type aliases */
    type RawRecord = Array[String]
    type PosFramePair = (PositionIndex, FrameIndex)
    type InitFile = (PosFramePair, os.Path)
    type IndexedRoi = DetectedRoi | SelectedRoi

    final case class CliConfig(
        beadRoisRoot: os.Path = null, // bogus, unconditionally required
        numShifting: PositiveInt = PositiveInt(1), // bogus, unconditionally required
        numAccuracy: PositiveInt = PositiveInt(1), // bogus, unconditionally required
        parserConfig: Option[os.Path] = None,
        outputFolder: Option[os.Path] = None, 
        optimisationTestingMode: Boolean = false, 
        referenceFrame: FrameIndex = FrameIndex(NonnegativeInt(Int.MaxValue)) // bogus, conditionally required
    )

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import parserBuilder.*
        import ScoptCliReaders.given

        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, VersionName), 
            opt[os.Path]("beadRoisRoot")
                .required()
                .action((p, c) => c.copy(beadRoisRoot = p))
                .validate(p => os.isDir(p).either(s"Alleged bead ROIs root isn't an extant folder: $p", ()))
                .text("Path to the folder with the detected bead ROIs"),
            opt[PositiveInt]("numShifting")
                .required()
                .action((n, c) => c.copy(numShifting = n))
                .text("Number of ROIs to use for shifting"),
            opt[PositiveInt]("numAccuracy")
                .required()
                .action((n, c) => c.copy(numAccuracy = n))
                .text("Number of ROIs to use for accuracy"),
            opt[os.Path]("parserConfig")
                .action((p, c) => c.copy(parserConfig = p.some))
                .validate(p => os.isFile(p).either(s"Alleged parser config isn't an extant file: $p", ()))
                .text("Path to parser configuration file, definining how to look for particular columns and fields"),
            opt[os.Path]('O', "outputFolder")
                .action((p, c) => c.copy(outputFolder = p.some))
                .text("Path to output root; if unspecified, use the input root."),
            opt[Unit]("optimisationTestingMode")
                .action((_, c) => c.copy(optimisationTestingMode = true))
                .text("Indicate that program's being run in optimisation R&D / testing mode, so tolerate insufficient ROIs")
                .children(
                    opt[NonnegativeInt]("referenceFrame")
                        .action((n, c) => c.copy(referenceFrame = FrameIndex(n)))
                        .text("0-based index of the frame/timepoint to be used as reference for drift correction")
                )
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => {
                val conf: os.Path | ParserConfig = opts.parserConfig.getOrElse(ParserConfig.default)
                workflow(
                    conf,
                    inputRoot = opts.beadRoisRoot, 
                    numShifting = opts.numShifting, 
                    numAccuracy = opts.numAccuracy, 
                    outputFolder = opts.outputFolder, 
                    referenceFrame = opts.optimisationTestingMode.option(opts.referenceFrame)
                    )
            }
        }
    }

    /* Business logic */
    def workflow(configFile: os.Path, inputRoot: os.Path, numShifting: PositiveInt, numAccuracy: PositiveInt): Unit = 
        workflow(configFile, inputRoot, numShifting, numAccuracy, None, None)
    
    def workflow(configFile: os.Path, inputRoot: os.Path, numShifting: PositiveInt, numAccuracy: PositiveInt, outputFolder: os.Path): Unit = 
        workflow(configFile, inputRoot, numShifting, numAccuracy, None, outputFolder.some)
    
    def workflow(
        conf: os.Path | ParserConfig, 
        inputRoot: os.Path, 
        numShifting: PositiveInt, 
        numAccuracy: PositiveInt, 
        referenceFrame: Option[FrameIndex],
        outputFolder: Option[os.Path]
        ): Unit = {

        val parserConfig = conf match {
            case pc: ParserConfig => pc
            case confFile: os.Path => 
                println(s"Reading parser config: ${confFile}")
                ParserConfig.readFileUnsafe(confFile)
        }
        
        /* Function definitions based on parsed config and CLI input */
        val writeRois = (rois: List[SelectedRoi], outpath: os.Path) => {
            println(s"Writing: $outpath")
            val jsonObjs = rois.map { r => SelectedRoi.toJsonSimple(parserConfig.coordinateSequence)(r) }
            os.makeDir.all(os.Path(outpath.toNIO.getParent))
            os.write.over(outpath, ujson.write(jsonObjs, indent = 4))
        }
        
        /* Actions */
        val inputFiles = discoverInputs(inputRoot)
        println(s"Input file count: ${inputFiles.size}")
        val outfolder = outputFolder.getOrElse(inputRoot)
        println(s"Will use output folder: $outfolder")
        // TODO: Any error should be fatal when reading ROIs file.
        // TODO: Only insufficient number of shifting ROIs should be fatal once partitioning.
        val (bads, goods): (List[(InitFile, RoiSplitFailure)], List[(InitFile, RoiSplitSuccess)]) = 
            Alternative[List].separate(
                preparePartitions(outfolder, parserConfig, numShifting = numShifting, numAccuracy = numAccuracy)(inputFiles.toList) map {
                    case (initFile, result) => PartitionAttempt.intoEither(result).bimap(initFile -> _, initFile -> _)
                }
            )
        if (bads.nonEmpty) {
            /* Check if we can tolerate (implicitly, by provision of reference frame) cases of too few ROIs.
             * If so, AND we have no parser errors (just too-few-ROI errors), AND none of the errors concerns 
             * the frame/timepoint designated as the reference, simply emit a warnings file rather than 
             * fatally crashing with an exception.
             */
            (referenceFrame, Alternative[List].separate(bads.map{ // Partition the list of problems by type of error.
                case kv@(_, _: RoisFileParseError) => kv.asLeft
                case ((pf, _), e: TooFewShiftingRois) => (pf, e).asRight
            })) match {
                case (Some(refFrame), (Nil, tooFewErrors)) if ( ! tooFewErrors.exists(_._1._2 === refFrame) ) => 
                    val warningsFile = outfolder / "roi_partition_warnings.severe.json"
                    println(s"Writing severe warnings file: $warningsFile")
                    os.write(warningsFile, write(tooFewErrors.map{ case (pf, tooFew) => pf -> tooFew.problem}, indent = 2))
                case _ => throw new Exception(s"${bads.size} (position, frame) pairs with problems.\n${bads}")
            }
        }
        val warnings: List[(PosFramePair, TooFewAccuracyRois)] = 
            goods flatMap { case (((p, f), _), splitResult) => 
                // Get the selected ROI groupings and build an optional warning.
                val (shifting, accuracy, warnOpt) = splitResult match {
                    case RoisPartition(shifting, accuracy) => (shifting, accuracy, None)
                    case tooFew: TooFewAccuracyRois => 
                        (tooFew.partition.shifting, tooFew.partition.accuracy, ((p, f), tooFew).some)
                }
                // Write the ROIs and emit the optional warning.
                writeRois(shifting, getOutputFilepath(outfolder)(p, f, Purpose.Shifting))
                writeRois(accuracy, getOutputFilepath(outfolder)(p, f, Purpose.Accuracy))
                warnOpt
            }
        if (warnings.isEmpty) then println("No warnings from bead ROIs partition, nice!")
        else {
            val warningsFile = outfolder / "roi_partition_warnings.json"
            println(s"Writing bead ROIs partition warnings file: $warningsFile")
            os.write(warningsFile, write(warnings.map{ case (pf, tooFew) => pf -> tooFew.problem }, indent = 2))
        }
        println("Done!")
    }

    given simplifiedTooFewRoisRW: ReadWriter[(PosFramePair, TooFewRois)] = readwriter[ujson.Value].bimap(
        pair => 
            val (pf, tooFew) = pair
            ujson.Obj(
                "position" -> ujson.Num(pf._1.get),
                "frame" -> ujson.Num(pf._2.get),
                "requested" -> ujson.Num(tooFew.requested),
                "realized" -> ujson.Num(tooFew.realized),
                "purpose" -> ujson.Str(tooFew.purpose.toString)
                ),
        json => 
            val pNel = Try{ PositionIndex.unsafe(json("position").int) }.toValidatedNel
            val fNel = Try{ FrameIndex.unsafe(json("frame").int) }.toValidatedNel
            val reqdNel = Try{ PositiveInt.unsafe(json("requested").int) }.toValidatedNel
            val realNel = Try{ NonnegativeInt.unsafe(json("realized").int) }.toValidatedNel
            val purposeNel = Try{ Purpose.valueOf(json("purpose").str) }.toValidatedNel
            (pNel, fNel, reqdNel, realNel, purposeNel).mapN(
                (p, f, reqd, real, purpose) => (p, f) -> TooFewRois(reqd, real, purpose)
            ).fold(errs => throw new Exception(f"${errs.size} error(s) reading simplified too-few-ROIs: ${errs.map(_.getMessage)}"), identity)
    )

    def writeTooFewRois[E : TooFewRoisLike] = (pf: PosFramePair, error: E) => {
        val tooFew = error.problem
        ujson.Obj(
            "position" -> ujson.Num(pf._1.get),
            "frame" -> ujson.Num(pf._2.get),
            "requested" -> ujson.Num(tooFew.requested),
            "realized" -> ujson.Num(tooFew.realized),
            "purpose" -> ujson.Str(tooFew.purpose.toString)
            )
    }

    def createParser(config: ParserConfig)(header: RawRecord): ErrMsgsOr[RawRecord => ErrMsgsOr[NonnegativeInt => DetectedRoi]] = {
        val maybeParseX = buildFieldParse(config.xCol.get, safeParseDouble.andThen(_.map(XCoordinate.apply)))(header)
        val maybeParseY = buildFieldParse(config.yCol.get, safeParseDouble.andThen(_.map(YCoordinate.apply)))(header)
        val maybeParseZ = buildFieldParse(config.zCol.get, safeParseDouble.andThen(_.map(ZCoordinate.apply)))(header)
        val maybeParseQC = buildFieldParse(config.qcCol, (s: String) => Right(s.isEmpty))(header)
        (maybeParseX, maybeParseY, maybeParseZ, maybeParseQC).mapN((x, y, z, qc) => (x, y, z, qc)).toEither.map{
            case (parseX, parseY, parseZ, parseQC) => { 
                (record: RawRecord) => (record.length === header.length)
                    .either(NEL.one(s"Header has ${header.length} fields but record has ${record.length}"), ())
                    .flatMap{ _ => 
                        val maybeX = parseX(record)
                        val maybeY = parseY(record)
                        val maybeZ = parseZ(record)
                        val maybeQC = parseQC(record)
                        (maybeX, maybeY, maybeZ, maybeQC).mapN((x, y, z, qcPass) => 
                            { (i: NonnegativeInt) => DetectedRoi(RoiIndex(i), Point3D(x, y, z), qcPass) }
                        ).toEither
                    }
            }
        }
    }

    def discoverInputs(inputsFolder: os.Path): Set[InitFile] = {
        def tryReadThruNN[A](f: NonnegativeInt => A): String => Option[A] = s => Try(s.toInt).toOption >>= NonnegativeInt.maybe.fmap(_.map(f))
        val prepFileMeta: os.Path => Option[InitFile] = filepath => {
            val filename = filepath.last
            if (filename.startsWith(BeadRoisPrefix)) {
                filename.split("\\.").head.stripPrefix(BeadRoisPrefix).split("_").toList match {
                    case "" :: rawPosIdx :: rawFrameIdx :: Nil => for {
                        position <- tryReadThruNN(PositionIndex.apply)(rawPosIdx)
                        frame <- tryReadThruNN(FrameIndex.apply)(rawFrameIdx)
                    } yield ((position, frame), filepath)
                    case _ => None
                }
            } else { None }
        }
        val results = os.list(inputsFolder).filter(os.isFile).toList.flatMap(prepFileMeta)
        val histogram = results.groupBy(_._1).filter(_._2.length > 1)
        if (histogram.nonEmpty) {
            given writeFiles: (Iterable[os.Path] => ujson.Value) with
                def apply(paths: Iterable[os.Path]) = paths.map(_.last)
            val errMsg = s"Non-unique filenames for key(s): ${posFrameMapToJson("filepaths", histogram.view.mapValues(_.map(_._2)).toMap)}"
            throw new IllegalStateException(errMsg)
        }
        results.toSet
    }
    
    def preparePartitions(outputFolder: os.Path, parserConfig: ParserConfig, numShifting: PositiveInt, numAccuracy: PositiveInt): 
        List[InitFile] => List[(InitFile, RoisFileParseError | RoisSplitResult)] = 
            _.map { case init => init -> readRoisFile(parserConfig)(init._2).fold(
                identity, 
                sampleDetectedRois(numShifting = numShifting, numAccuracy = numAccuracy))
            }

    /**
      * Read a single (one FOV, one frame) ROIs file.
      * 
      * Potential "failures":
      * 1. Given path isn't a file
      * 2. Field delimiter can't be inferred from given path's extension
      * 3. Given file is empty
      * 4. One or more columns required (by the parserConfig) to parse aren't in file'e header (first line)
      * 5. Any record fails to parse
      * 
      * @param parserConfig How to parse the file
      * @param roisFile The file to parse
      * @return A collection of ROIs, representing what was detected for a particular (FOV, frame) combo
      */
    def readRoisFile(parserConfig: ParserConfig)(roisFile: os.Path): Either[RoisFileParseError, Iterable[DetectedRoi]] = {
        prepFileRead(roisFile)
            .toEither
            .flatMap{ case (sep, head, lines) => createParser(parserConfig)(sep `split` head).map(_ -> lines.map(sep.split)) }
            .leftMap(RoisFileParseFailedSetup.apply)
            .flatMap { case (parse, rawRecords) => 
                Alternative[List].separate(
                    NonnegativeInt.indexed(rawRecords)
                        .map{ case (rr, i) => parse(rr).bimap(errs => BadRecord(i, rr, errs), _(i)) }
                ) match { case (bads, rois) => bads.toNel.toLeft(rois).leftMap(RoisFileParseFailedRecords.apply) }
            }
    }

    /**
      * Try the given ROIs into a pool for actual drift correction and a (nonoverlapping) pool for accuracy assessment.
      *
      * There are 3 output cases, based on count of usable ROIs...
      * 1. ...exceeds or equals the sum of shifting and accuracy, all good
      * 2. ...exceeds or equals shifting count, but not enough to "fill the order" for accuracy, yielding partiting with warning
      * 3. ...less than shifting count, yielding an error-like type with no partition
      * 
      * @param numShifting Number of ROIs to use for actual drift correction
      * @param numAccuracy Number of ROIs to use for drift correction accuracy assessment
      * @param rois Collection of detected bead ROIs, from a single (FOV, frame) pair
      * @return An explanation of failure if partition isn't possible, or a partition with perhaps a warning
      */
    def sampleDetectedRois(numShifting: PositiveInt, numAccuracy: PositiveInt)(rois: Iterable[DetectedRoi]): RoisSplitResult = {
        val sampleSize = numShifting + numAccuracy
        if (sampleSize < numShifting || sampleSize < numAccuracy) {
            val msg = s"Appears overflow occurred computing sample size: ${numShifting} + ${numAccuracy} = ${sampleSize}"
            throw new IllegalArgumentException(msg)
        }
        
        val pool = rois.filter(_.isUsable)
        val (inSample, _) = Random.shuffle(pool.toList) `splitAt` sampleSize
        val (shifting, remaining) = inSample `splitAt` numShifting
        val accuracy = remaining `take` numAccuracy
        
        val shiftingRealized = NonnegativeInt.unsafe(shifting.length)
        if (shiftingRealized < numShifting) TooFewShiftingRois(numShifting, shiftingRealized)
        else {
            val partition = RoisPartition(
                shifting.map(roi => RoiForShifting(roi.index, roi.centroid)), 
                accuracy.map(roi => RoiForAccuracy(roi.index, roi.centroid))
                )
            if (accuracy.length < numAccuracy) TooFewAccuracyRois(partition, numAccuracy, accuracy.length) else partition
        }
    }

    /***********************/
    /* Helper types        */
    /***********************/
    sealed trait RoisFileParseError extends Throwable
    final case class RoisFileParseFailedSetup(get: ErrorMessages) extends RoisFileParseError
    final case class RoisFileParseFailedRecords(get: NEL[BadRecord]) extends RoisFileParseError
    
    type RoiSplitFailure = RoisFileParseError | TooFewShiftingRois
    type RoiSplitOutcome = Either[RoiSplitFailure, RoiSplitSuccess]

    sealed trait RoisSplitResult
    sealed trait RoiSplitSuccess extends RoisSplitResult:
        def partition: RoisPartition
    
    final case class TooFewShiftingRois(requested: PositiveInt, realized: NonnegativeInt) extends RoisSplitResult
    object TooFewShiftingRois:
        given TooFewForShifting: TooFewRoisLike[TooFewShiftingRois] with
            def getTooFew = { case TooFewShiftingRois(requested, realized) => TooFewRois(requested, realized, Purpose.Shifting) }
    
    final case class TooFewAccuracyRois(partition: RoisPartition, requested: PositiveInt, realized: Int) extends RoiSplitSuccess
    object TooFewAccuracyRois:
        given TooFewForAccuracy: TooFewRoisLike[TooFewAccuracyRois] with
            def getTooFew = { case TooFewAccuracyRois(_, requested, realized) => TooFewRois(requested, realized, Purpose.Accuracy) }
    
    final case class RoisPartition(shifting: List[RoiForShifting], accuracy: List[RoiForAccuracy]) extends RoiSplitSuccess:
        final def partition = this
    
    final case class TooFewRois(requested: PositiveInt, realized: Int, purpose: Purpose) derives ReadWriter:
        require(requested > realized, s"Count of realized ROIs ($realized) isn't less than count of requested ($requested)")
    
    /** A type which admits a {@code TooFewRois} value */
    trait TooFewRoisLike[A]:
        def getTooFew: A => TooFewRois

    extension [A](a: A)(using ev: TooFewRoisLike[A])
        def problem = ev.getTooFew(a)

    object RoisSplitResult:
        def intoEither = (_: RoisSplitResult) match {
            case p: RoisPartition => p.asRight
            case a: TooFewAccuracyRois => a.asRight
            case s: TooFewShiftingRois => s.asLeft
        }
    end RoisSplitResult

    object PartitionAttempt:
        def intoEither(eOrR: RoisFileParseError | RoisSplitResult): RoiSplitOutcome = eOrR match {
            case e: RoisFileParseError => e.asLeft
            case r: RoisSplitResult => RoisSplitResult.intoEither(r)
        }
    end PartitionAttempt

    sealed trait ColumnName { def get: String }
    final case class XColumn(get: String) extends ColumnName
    final case class YColumn(get: String) extends ColumnName
    final case class ZColumn(get: String) extends ColumnName
    
    object ColumnName:
        given toJson(using liftStr: String => ujson.Value): (ColumnName => ujson.Value) = cn => liftStr(cn.get)

    final case class BadRecord(index: NonnegativeInt, record: List[String], problems: ErrorMessages)
    object BadRecord:
        def apply(index: NonnegativeInt, record: RawRecord, problems: ErrorMessages) = new BadRecord(index, record.toList, problems)

    /** How the ROIs will be used w.r.t. drift correction (either actual shift, or accuracy assessment) */
    enum Purpose derives ReadWriter:
        case Shifting, Accuracy
        def lowercase = this.toString.toLowerCase
    
    final case class Filename(get: String)
    
    final case class ParserConfig(xCol: XColumn, yCol: YColumn, zCol: ZColumn, qcCol: String, coordinateSequence: CoordinateSequence):
        {
            val textLikeMembers = Map("x" -> xCol.get, "y" -> yCol.get, "z" -> zCol.get, "qc" -> qcCol)
            textLikeMembers.toList.groupBy(_._2).view.mapValues(_.map(_._1)).toList.filter(_._2.length > 1) match {
                case Nil => ()
                case reps => throw new IllegalArgumentException(
                    s"${reps.size} repeat(s) among text-like columns for parsing config: ${reps mkString ","}")
            }
        }

    /** Helpers for working with the parser configuration */
    object ParserConfig:
        val (xFieldName, yFieldName, zFieldName, qcFieldName, csFieldName) = labelsOf[ParserConfig]
        
        given jsonCodec: ReadWriter[ParserConfig] = readwriter[ujson.Value].bimap(
            pc => ujson.Obj(
                xFieldName -> ColumnName.toJson(pc.xCol), 
                yFieldName -> ColumnName.toJson(pc.yCol), 
                zFieldName -> ColumnName.toJson(pc.zCol), 
                qcFieldName -> ujson.Str(pc.qcCol), 
                csFieldName -> CoordinateSequence.toJson(pc.coordinateSequence)
            ), 
            json => {
                val xNel = safeExtract(xFieldName, XColumn.apply)(json)
                val yNel = safeExtract(yFieldName, YColumn.apply)(json)
                val zNel = safeExtract(zFieldName, ZColumn.apply)(json)
                val qcNel = safeExtract(qcFieldName, identity)(json)
                val csNel = safeExtract(csFieldName, CoordinateSequence.fromJsonSafe(_).fold(throw _, identity))(json)
                (xNel, yNel, zNel, qcNel, csNel).mapN(ParserConfig.apply).fold(errs => throw new ParseError(errs), identity)
            }
        )

        /** A default value corresponding to what we have in looptrace Python code */
        def default = ParserConfig(
            XColumn("centroid-2"), 
            YColumn("centroid-1"), 
            ZColumn("centroid-0"), 
            qcCol = "fail_code",
            coordinateSequence = CoordinateSequence.Reverse  // WRITE the coordinates in (z, y, x) order to JSON.
        )
        
        def readFile(confFile: os.Path): Either[ParseError, ParserConfig] = 
            Try{ UJsonHelpers.readJsonFile[ParserConfig](confFile) }
                .toEither
                .leftMap{
                    case e: ParseError => e
                    case e: java.nio.file.NoSuchFileException => ParseError(NEL.one(s"Alleged config path isn't a file: $confFile"))
                    case e: ujson.IncompleteParseException => ParseError(NEL.one(s"Empty config file ($confFile)? Got error: ${e.getMessage}"))
                }

        def readFileUnsafe(confFile: os.Path): ParserConfig = readFile(confFile).fold(throw _, identity)

        final case class ParseError(errorMessages: NEL[String]) extends Exception(s"${errorMessages.size} errors: ${errorMessages}")
    end ParserConfig

    /* Helper functions */
    def getOutputFilename(pos: PositionIndex, frame: FrameIndex, purpose: Purpose): Filename =
        Filename(s"${BeadRoisPrefix}_${pos.get}_${frame.get}.${purpose.lowercase}.json")

    def getOutputSubfolder(root: os.Path) = root / (_: Purpose).lowercase

    def getOutputFilepath(root: os.Path)(pos: PositionIndex, frame: FrameIndex, purpose: Purpose): os.Path = 
        getOutputSubfolder(root)(purpose) / getOutputFilename(pos, frame, purpose).get

    private def prepFileRead(roisFile: os.Path): ValidatedNel[String, (Delimiter, String, List[String])] = {
        val maybeSep = Delimiter.fromPath(roisFile).toRight(s"Cannot infer delimiter for file! $roisFile").toValidatedNel
        val maybeHeadTail = (os.read.lines(roisFile).toList match {
            case Nil => Left(f"No lines in file! $roisFile")
            case h :: t => Right(h -> t)
        }).toValidatedNel
        (maybeSep, maybeHeadTail).mapN{ case (sep, (h, t)) => (sep, h, t) }
    }

end PartitionIndexedDriftCorrectionRois