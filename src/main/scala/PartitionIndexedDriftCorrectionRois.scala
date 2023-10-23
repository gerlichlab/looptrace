package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ Random, Try }
import cats.Alternative
import cats.data.{ NonEmptyList as NEL, ValidatedNel }
import cats.syntax.apply.*
import cats.syntax.either.*
import cats.syntax.flatMap.*
import cats.syntax.functor.*
import cats.syntax.list.*
import cats.syntax.option.*
import mouse.boolean.*
import upickle.default.*
import scopt.OParser

import space.CoordinateSequence
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.PartitionIndexedPoints.RawRecord

/** Split pool of detected bead ROIs into those for drift correction shift, drift correction accuracy, and unused. */
object PartitionIndexedPoints {
    val ProgramName = "PartitionDriftCorrectionRois"

    val BeadRoisPrefix = "bead_rois__"

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
        outputFolder: Option[os.Path] = None,
        shiftingSubfolderName: String = "shifting",
        accuracySubfolderName: String = "accuracy",
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
                .action((p, c) => c.copy(outputFolder = p.some)),
            opt[String]("shiftingSubfolderName")
                .action((fn, c) => c.copy(shiftingSubfolderName = fn))
                .text("Name of the subfolder in which to place the ROIs selected for drift correction shifting"),
            opt[String]("accuracySubfolderName")
                .action((fn, c) => c.copy(accuracySubfolderName = fn))
                .text("Name of the subfolder in which to place the ROIs selected for drift correction accuracy"),
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => {
                
                /* Function definitions based on raw CLI inputs */
                // TODO: try to restrict this to just the specific subtype, so that the ROIs collection can't be mixed.
                val partition = sampleDetectedRois(numShifting = opts.numShifting, numAccuracy = opts.numAccuracy).fmap(_.leftMap(NEL.one))
                
                /* Configuration of input parser */
                println(s"Reading parser config: ${opts.parserConfig}")
                val parserConfig: ParserConfig = readJson[ParserConfig](opts.parserConfig)
                
                /* Function definitions based on parsed config and CLI input */
                val writeRois = (rois: NEL[SelectedRoi], outpath: os.Path) => {
                    println(s"Writing: $outpath")
                    val jsonObjs = rois.toList map { r => SelectedRoi.toJsonSimple(parserConfig.coordinateSequence)(r) }
                    os.makeDir.all(os.Path(outpath.toNIO.getParent))
                    os.write.over(outpath, ujson.write(jsonObjs, indent = 4))
                }
                
                /* Actions */
                val inputFiles = discoverInputs(opts.beadRoisRoot)
                println(s"Input file count: ${inputFiles.length}")
                val outfolder = opts.outputFolder.getOrElse(opts.beadRoisRoot)
                println(s"Will use output folder: $outfolder")
                val (bads, goods) = preparePartitions(outfolder, parserConfig, numShifting = opts.numShifting, numAccuracy = opts.numAccuracy)(inputFiles)
                if (bads.nonEmpty) throw new Exception(s"${bads.length} (position, frame) pairs with problems.\n${bads}")
                goods foreach { case ((shiftingRois, getShiftingPath), (accuracyRois, getAccuracyPath)) => 
                    writeRois(shiftingRois, getShiftingPath(outfolder))
                    writeRois(accuracyRois, getAccuracyPath(outfolder))
                }
            }
        }
    }

    /* Business logic */
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
        val maybeParseX: ValidatedNel[String, RawRecord => ValidatedNel[String, XCoordinate]] = getRawFieldParse(config.xCol.get)(safeParseDouble.andThen(_.map(XCoordinate.apply)))
        val maybeParseY = getRawFieldParse(config.yCol.get)(safeParseDouble.andThen(_.map(YCoordinate.apply)))
        val maybeParseZ = getRawFieldParse(config.zCol.get)(safeParseDouble.andThen(_.map(ZCoordinate.apply)))
        val maybeParseQC = getRawFieldParse(config.qcCol)((s: String) => Right(s.isEmpty))
        (maybeParseX, maybeParseY, maybeParseZ, maybeParseQC).mapN((x, y, z, qc) => (x, y, z, qc)).toEither.map{
            case (parseX, parseY, parseZ, parseQC) => { (record: RawRecord) => 
                val maybeX = parseX(record)
                val maybeY = parseY(record)
                val maybeZ = parseZ(record)
                val maybeQC = parseQC(record)
                (maybeX, maybeY, maybeZ, maybeQC).mapN((x, y, z, qcPass) => { (i: NonnegativeInt) => DetectedRoi(RoiIndex(i), Point3D(x, y, z), qcPass) }).toEither
            }
        }
    }

    def discoverInputs(inputsFolder: os.Path): List[InitFile] = {
        def tryReadThruNN[A](f: NonnegativeInt => A): String => Option[A] = s => Try(s.toInt).toOption >>= NonnegativeInt.maybe.fmap(_.map(f))
        val prepFileMeta: os.Path => Option[InitFile] = filepath => {
            filepath.last.split("\\.").head.stripPrefix(BeadRoisPrefix).split("_").toList match {
                case rawPosIdx :: rawFrameIdx :: Nil => for {
                    position <- tryReadThruNN(PositionIndex.apply)(rawPosIdx)
                    frame <- tryReadThruNN(FrameIndex.apply)(rawFrameIdx)
                } yield (position, frame, filepath)
                case _ => None
            }
        }
        os.list(inputsFolder).filter(os.isFile).toList.flatMap(prepFileMeta)
    }
    
    def preparePartitions(
        outputFolder: os.Path, 
        parserConfig: ParserConfig, 
        numShifting: PositiveInt, 
        numAccuracy: PositiveInt
        )(inputs: List[InitFile]): (List[NEL[String] | NEL[BadRecord]], List[((NEL[SelectedRoi], os.Path => os.Path), (NEL[SelectedRoi], os.Path => os.Path))]) = {
        val partition = sampleDetectedRois(numShifting = numShifting, numAccuracy = numAccuracy).fmap(_.leftMap(NEL.one))
        val getErrMsg = (purpose: String, pos: PositionIndex, frame: FrameIndex) => s"No ROIs for $purpose in (${pos.get}, ${frame.get})"
        def tryPrep(purpose: String, pos: PositionIndex, frame: FrameIndex, rois: List[SelectedRoi]): ErrMsgsOr[(NEL[SelectedRoi], os.Path => os.Path)] = 
            tryPrepRois(pos, frame, rois).toRight{ NEL.one(getErrMsg(purpose, pos, frame)) }
        Alternative[List].separate(inputs.map { case (pos, frame, roiFile) => 
            for {
                rois <- readFile(parserConfig)(roiFile)
                (_, shiftingRois, accuracyRois) <- partition(rois)
                shifting <- tryPrep("shifting", pos, frame, shiftingRois)
                accuracy <- tryPrep("accuracy", pos, frame, accuracyRois)
            } yield (shifting, accuracy)
        })
    }
    
    def readFile(parserConfig: ParserConfig)(roisFile: os.Path): Either[ErrorMessages | NEL[BadRecord], Iterable[DetectedRoi]] = {
        val maybeParserAndRecords: ErrMsgsOr[(RawRecord => ErrMsgsOr[NonnegativeInt => DetectedRoi], List[RawRecord])] = for {
            (sep, head, lines) <- prepFileRead(roisFile).toEither
            safeParse <- createParser(parserConfig)(sep.split(head, -1))
        } yield (safeParse, lines map (sep.split(_, -1)))
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
    sealed trait ColumnName
    final case class XColumn(get: String) extends ColumnName
    final case class YColumn(get: String) extends ColumnName
    final case class ZColumn(get: String) extends ColumnName
    
    object ColumnName:
        given rwXcol: ReadWriter[XColumn] = readwriter[String].bimap(_.get, XColumn(_)) // Facilitate RW[ParserConfig] derivation.
        given rwYcol: ReadWriter[YColumn] = readwriter[String].bimap(_.get, YColumn(_)) // Facilitate RW[ParserConfig] derivation.
        given rwZcol: ReadWriter[ZColumn] = readwriter[String].bimap(_.get, ZColumn(_)) // Facilitate RW[ParserConfig] derivation.

    final case class BadRecord(index: NonnegativeInt, record: RawRecord, problems: ErrorMessages)
    final case class NameOfPurpose(get: String)
    final case class Filename(get: String)
    
    import ColumnName.given
    final case class ParserConfig(xCol: XColumn, yCol: YColumn, zCol: ZColumn, qcCol: String, coordinateSequence: CoordinateSequence) derives ReadWriter:
        {
            val textLikeMembers: List[String] = List(xCol.get, yCol.get, zCol.get, qcCol)
            textLikeMembers.groupBy(identity).withFilter(_._2.length > 1).map(_._1) match {
                case Nil => ()
                case reps => throw new IllegalArgumentException(
                    s"${reps.size} repeats among text-like columns for parsing config: ${reps mkString ","}")
            }
        }

    /* Helper functions */
    def getOutputFilename(pos: PositionIndex, frame: FrameIndex, purpose: NameOfPurpose): Filename = {
        Filename(s"${BeadRoisPrefix}${pos.get}_${frame.get}.${purpose.get}.json")
    }

    def tryPrepRois(position: PositionIndex, frame: FrameIndex, rois: List[SelectedRoi]): Option[(NEL[SelectedRoi], os.Path => os.Path)] = rois match {
        case Nil => None
        case firstRoi :: remainingRois => {
            val nameOfPurpose = NameOfPurpose(firstRoi match {
                case _: RoiForShifting => "shifting"
                case _: RoiForAccuracy => "accuracy"
            })
            val filename = getOutputFilename(position, frame, nameOfPurpose)
            (NEL(firstRoi, remainingRois), (_: os.Path) / nameOfPurpose.get / filename.get).some
        }
    }

    def prepFileRead(roisFile: os.Path): ValidatedNel[String, (Delimiter, String, List[String])] = {
        val maybeSep = Delimiter.infer(roisFile).toRight(f"Cannot infer delimiter for file: $roisFile").toValidatedNel
        val maybeHeadTail = (os.read.lines(roisFile).toList match {
            case Nil => Left(f"No lines in file! $roisFile")
            case h :: t => Right(h -> t)
        }).toValidatedNel
        (maybeSep, maybeHeadTail).mapN{ case (sep, (h, t)) => (sep, h, t) }
    }
    
    def safeParseDouble(s: String): Either[String, Double] = Try{ s.toDouble }.toEither.leftMap(_.getMessage)
}
