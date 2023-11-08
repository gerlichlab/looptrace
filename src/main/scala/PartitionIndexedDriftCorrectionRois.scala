package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ Random, Try }
import cats.Alternative
import cats.data.{ NonEmptyList as NEL, ValidatedNel }
import cats.syntax.apply.*
import cats.syntax.either.*
import cats.syntax.eq.*
import cats.syntax.flatMap.*
import cats.syntax.functor.*
import cats.syntax.list.*
import cats.syntax.option.*
import cats.syntax.validated.*
import mouse.boolean.*
import upickle.default.*
import scopt.OParser

import space.CoordinateSequence
import at.ac.oeaw.imba.gerlich.looptrace.space.*

/** Split pool of detected bead ROIs into those for drift correction shift, drift correction accuracy, and unused. */
object PartitionIndexedDriftCorrectionRois {
    val ProgramName = "PartitionIndexedDriftCorrectionRois"

    val BeadRoisPrefix = "bead_rois_"

    /* Type aliases */
    type RawRecord = Array[String]
    type ErrorMessages = NEL[String]
    type ErrMsgsOr[A] = Either[ErrorMessages, A]
    type InitFile = (PositionIndex, FrameIndex, os.Path)
    type IndexedRoi = DetectedRoi | SelectedRoi

    final case class CliConfig(
        parserConfig: os.Path = null,
        beadRoisRoot: os.Path = null,
        numShifting: PositiveInt = PositiveInt(1),
        numAccuracy: PositiveInt = PositiveInt(1),
        outputFolder: Option[os.Path] = None
    )

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import parserBuilder.*
        import CliReaders.given

        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, VersionName), 
            opt[os.Path]("parserConfig")
                .required()
                .action((p, c) => c.copy(parserConfig = p))
                .validate(p => os.isFile(p).either(s"Alleged parser config isn't an extant file: $p", ()))
                .text("Path to parser configuration file, definining how to look for particular columns and fields"),
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
            opt[os.Path]('O', "outputFolder")
                .action((p, c) => c.copy(outputFolder = p.some))
                .text("Path to output root; if unspecified, use the input root.")
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => workflow(
                configFile = opts.parserConfig, 
                inputRoot = opts.beadRoisRoot, 
                numShifting = opts.numShifting, 
                numAccuracy = opts.numAccuracy, 
                outputFolder = opts.outputFolder
                )
        }
    }

    /* Business logic */
    def workflow(configFile: os.Path, inputRoot: os.Path, numShifting: PositiveInt, numAccuracy: PositiveInt): Unit = 
        workflow(configFile, inputRoot, numShifting, numAccuracy, None)
    def workflow(configFile: os.Path, inputRoot: os.Path, numShifting: PositiveInt, numAccuracy: PositiveInt, outputFolder: os.Path): Unit = 
        workflow(configFile, inputRoot, numShifting, numAccuracy, outputFolder.some)
    def workflow(configFile: os.Path, inputRoot: os.Path, numShifting: PositiveInt, numAccuracy: PositiveInt, outputFolder: Option[os.Path]): Unit = {
        /* Configuration of input parser */
        println(s"Reading parser config: ${configFile}")
        val parserConfig = ParserConfig.readFileUnsafe(configFile)
        
        /* Function definitions based on parsed config and CLI input */
        val writeRois = (rois: NEL[SelectedRoi], outpath: os.Path) => {
            println(s"Writing: $outpath")
            val jsonObjs = rois.toList map { r => SelectedRoi.toJsonSimple(parserConfig.coordinateSequence)(r) }
            os.makeDir.all(os.Path(outpath.toNIO.getParent))
            os.write.over(outpath, ujson.write(jsonObjs, indent = 4))
        }
        
        /* Actions */
        val inputFiles = discoverInputs(inputRoot)
        println(s"Input file count: ${inputFiles.size}")
        val outfolder = outputFolder.getOrElse(inputRoot)
        println(s"Will use output folder: $outfolder")
        val (bads, goods) = preparePartitions(outfolder, parserConfig, numShifting = numShifting, numAccuracy = numAccuracy)(inputFiles.toList)
        if (bads.nonEmpty) throw new Exception(s"${bads.length} (position, frame) pairs with problems.\n${bads}")
        goods foreach { case ((shiftingRois, getShiftingPath), (accuracyRois, getAccuracyPath)) => 
            writeRois(shiftingRois, getShiftingPath(outfolder))
            writeRois(accuracyRois, getAccuracyPath(outfolder))
        }
    }

    def createParser(config: ParserConfig)(header: RawRecord): ErrMsgsOr[RawRecord => ErrMsgsOr[NonnegativeInt => DetectedRoi]] = {
        def getRawFieldParse[A](name: String)(parse: String => Either[String, A]): ValidatedNel[String, RawRecord => ValidatedNel[String, A]] = {
            header.indexOf(name) match {
                case -1 => Left(f"Missing field in header: $name").toValidatedNel
                case i => Right{ (rr: RawRecord) => Try{ rr(i) }
                    .toEither
                    .leftMap(_ => s"Out of bounds finding value for '$name' in record with ${rr.length} fields: $i")
                    .flatMap(parse)
                    .toValidatedNel
                }.toValidatedNel
            }
        }
        val maybeParseX = getRawFieldParse(config.xCol.get)(safeParseDouble.andThen(_.map(XCoordinate.apply)))
        val maybeParseY = getRawFieldParse(config.yCol.get)(safeParseDouble.andThen(_.map(YCoordinate.apply)))
        val maybeParseZ = getRawFieldParse(config.zCol.get)(safeParseDouble.andThen(_.map(ZCoordinate.apply)))
        val maybeParseQC = getRawFieldParse(config.qcCol)((s: String) => Right(s.isEmpty))
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
                    } yield (position, frame, filepath)
                    case _ => None
                }
            } else { None }
        }
        val results = os.list(inputsFolder).filter(os.isFile).toList.flatMap(prepFileMeta)
        val histogram = results.groupBy{ case (pos, frame, _) => pos -> frame }.filter(_._2.length > 1)
        if (histogram.nonEmpty) {
            given writeFiles: (Iterable[os.Path] => ujson.Value) with
                def apply(paths: Iterable[os.Path]) = paths.map(_.last)
            val errMsg = s"Non-unique filenames for key(s): ${posFrameMapToJson("filepaths", histogram.view.mapValues(_.map(_._3)).toMap)}"
            throw new IllegalStateException(errMsg)
        }
        results.toSet
    }
    
    def preparePartitions(
        outputFolder: os.Path, 
        parserConfig: ParserConfig, 
        numShifting: PositiveInt, 
        numAccuracy: PositiveInt
        )(inputs: List[InitFile]): (List[NEL[String] | NEL[BadRecord]], List[((NEL[SelectedRoi], os.Path => os.Path), (NEL[SelectedRoi], os.Path => os.Path))]) = {
        // TODO: try to restrict this to just the specific subtype, so that the ROIs collection can't be mixed.
        val partition = sampleDetectedRois(numShifting = numShifting, numAccuracy = numAccuracy).fmap(_.leftMap(NEL.one))
        val getErrMsg = (purpose: String, pos: PositionIndex, frame: FrameIndex) => s"No ROIs for $purpose in (${pos.get}, ${frame.get})"
        def tryPrep(purpose: String, pos: PositionIndex, frame: FrameIndex, rois: List[SelectedRoi]): ErrMsgsOr[(NEL[SelectedRoi], os.Path => os.Path)] = 
            tryPrepRois(pos, frame, rois).toRight{ NEL.one(getErrMsg(purpose, pos, frame)) }
        Alternative[List].separate(inputs.map { case (pos, frame, roiFile) => 
            // Any error should be fatal when reading ROIs file.
            // Only insufficient number of shifting ROIs should be fatal once partitioning.
            readRoisFile(parserConfig)(roiFile).flatMap(partition).flatMap{ case (_, shiftingRois, accuracyRois) => 
                val shiftingValidNel = tryPrep("shifting", pos, frame, shiftingRois).toValidated
                val accuracyValidNel = tryPrep("accuracy", pos, frame, accuracyRois).toValidated
                (shiftingValidNel, accuracyValidNel).mapN((_1, _2) => (_1, _2)).toEither
            }
        })
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
    def readRoisFile(parserConfig: ParserConfig)(roisFile: os.Path): Either[ErrorMessages | NEL[BadRecord], Iterable[DetectedRoi]] = {
        val maybeParserAndRecords: ErrMsgsOr[(RawRecord => ErrMsgsOr[NonnegativeInt => DetectedRoi], List[RawRecord])] = for {
            (sep, head, lines) <- prepFileRead(roisFile).toEither
            safeParse <- createParser(parserConfig)(sep `split` head)
        } yield (safeParse, lines map sep.split)
        maybeParserAndRecords flatMap { case (parse, rawRecords) => 
            val (bads, rois) = Alternative[List].separate(
                NonnegativeInt.indexed(rawRecords).map{ case (rr, i) => parse(rr).bimap(errs => BadRecord(i, rr, errs), _(i)) } )
            bads.toNel.toLeft(rois)
        }
    }

    def sampleDetectedRois(numShifting: PositiveInt, numAccuracy: PositiveInt)(rois: Iterable[DetectedRoi]): Either[String, (List[DetectedRoi], List[RoiForShifting], List[RoiForAccuracy])] = {
        val sampleSize = numShifting + numAccuracy
        if (sampleSize < numShifting || sampleSize < numAccuracy) {
            val msg = s"Appears overflow occurred: ${numShifting} + ${numAccuracy} = ${sampleSize}"
            throw new IllegalArgumentException(msg)
        }
        val pool = rois.filter(_.isUsable)
        val numAvailable = pool.size
        (sampleSize <= numAvailable).either(s"Fewer ROIs available than requested! $numAvailable < $sampleSize", {
            val (inSample, outOfSample) = Random.shuffle(pool.toList).splitAt(sampleSize)
            val (forShifting, forAccuracy) = inSample.splitAt(numShifting)
            val shiftingRois = forShifting.map(roi => RoiForShifting(roi.index, roi.centroid))
            val accuracyRois = forAccuracy.map(roi => RoiForAccuracy(roi.index, roi.centroid))
            (outOfSample, shiftingRois, accuracyRois)
        })
    }

    /* Helper types */
    sealed trait ColumnName { def get: String }
    final case class XColumn(get: String) extends ColumnName
    final case class YColumn(get: String) extends ColumnName
    final case class ZColumn(get: String) extends ColumnName
    
    object ColumnName:
        given toJson(using liftStr: String => ujson.Value): (ColumnName => ujson.Value) = cn => liftStr(cn.get)

    final case class BadRecord(index: NonnegativeInt, record: List[String], problems: ErrorMessages)
    object BadRecord:
        def apply(index: NonnegativeInt, record: RawRecord, problems: ErrorMessages) = new BadRecord(index, record.toList, problems)

    final case class NameOfPurpose(get: String)
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

    object ParserConfig:
        val (xFieldName, yFieldName, zFieldName, qcFieldName, csFieldName) = labelsOf[ParserConfig]
        
        private def safeExtract[A](key: String, lift: String => A)(json: ujson.Value): ValidatedNel[String, A] = 
            Try{ json(key).str }.toEither.bimap(_.getMessage, lift).toValidatedNel
        
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
    def getOutputFilename(pos: PositionIndex, frame: FrameIndex, purpose: NameOfPurpose): Filename = {
        Filename(s"${BeadRoisPrefix}_${pos.get}_${frame.get}.${purpose.get}.json")
    }

    private def tryPrepRois(position: PositionIndex, frame: FrameIndex, rois: List[SelectedRoi]): Option[(NEL[SelectedRoi], os.Path => os.Path)] = rois.toNel.map{ rs =>
        val nameOfPurpose = NameOfPurpose(rs.head match {
            case _: RoiForShifting => "shifting"
            case _: RoiForAccuracy => "accuracy"
        })
        val filename = getOutputFilename(position, frame, nameOfPurpose)
        (rs, (_: os.Path) / nameOfPurpose.get / filename.get)
    }

    private def prepFileRead(roisFile: os.Path): ValidatedNel[String, (Delimiter, String, List[String])] = {
        val maybeSep = Delimiter.fromPath(roisFile).toRight(s"Cannot infer delimiter for file! $roisFile").toValidatedNel
        val maybeHeadTail = (os.read.lines(roisFile).toList match {
            case Nil => Left(f"No lines in file! $roisFile")
            case h :: t => Right(h -> t)
        }).toValidatedNel
        (maybeSep, maybeHeadTail).mapN{ case (sep, (h, t)) => (sep, h, t) }
    }
    
    private def safeParseDouble(s: String): Either[String, Double] = Try{ s.toDouble }.toEither.leftMap(_.getMessage)
}
