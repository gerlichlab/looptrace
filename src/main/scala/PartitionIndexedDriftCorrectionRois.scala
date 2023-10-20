package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ Random, Try }
import cats.Alternative
import cats.data.{ NonEmptyList as NEL, ValidatedNel }
import cats.syntax.apply.*
import cats.syntax.either.*
import cats.syntax.flatMap.*
import cats.syntax.functor.*
import cats.syntax.list.*
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

    final case class CliConfig(
        parserConfig: os.Path = null,
        beadRoisRoot: os.Path = null,
        numShifting: PositiveInt = PositiveInt(1),
        numAccuracy: PositiveInt = PositiveInt(1),
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
                // TODO: try to restrict this to just the specific subtype, so that the ROIs collection can't be mixed.
                def getRoisOutfile[R <: IndexedRoi](position: PositionIndex, frame: FrameIndex, rois: List[R]): Option[(NEL[R], os.Path)] = rois.toNel.map{ rois =>
                    val nameOfPurpose = rois.head match {
                        case _: DetectedRoi => "unused"
                        case _: RoiForShifting => "shifting"
                        case _: RoiForAccuracy => "accuracy"
                    }
                    val filename = s"${BeadRoisPrefix}${position.get}_${frame.get}.${nameOfPurpose}.json"
                    val outfile = opts.beadRoisRoot / nameOfPurpose / filename
                    rois -> outfile
                }

                println(s"Reading parser config: ${opts.parserConfig}")
                val parserConfig: ParserConfig = read[ParserConfig](os.read(opts.parserConfig))
                val partition = sampleDetectedRois(numShifting = opts.numShifting, numAccuracy = opts.numAccuracy).fmap(_.leftMap(NEL.one))
                def tryReadThruNN[A](f: NonnegativeInt => A): String => Option[A] = s => Try(s.toInt).toOption >>= NonnegativeInt.maybe.fmap(_.map(f))
                val prepFileMeta: os.Path => Option[(PositionIndex, FrameIndex, os.Path)] = filepath => {
                    filepath.last.split("\\.").head.stripPrefix(BeadRoisPrefix).split("_").toList match {
                        case rawPosIdx :: rawFrameIdx :: Nil => for {
                            position <- tryReadThruNN(PositionIndex.apply)(rawPosIdx)
                            frame <- tryReadThruNN(FrameIndex.apply)(rawFrameIdx)
                        } yield (position, frame, filepath)
                        case _ => None
                    }
                }
                val inputFiles = os.list(opts.beadRoisRoot).filter(os.isFile).toList.flatMap(prepFileMeta)
                println(s"Input file count: ${inputFiles.length}")
                val (bads, goods) = Alternative[List].separate(inputFiles.map { case (pos, frame, roiFile) => 
                    val maybeOutputPaths = for {
                        rois <- readFile(parserConfig)(roiFile)
                        (_, shiftingRois, accuracyRois) <- partition(rois)
                        shifting <- getRoisOutfile(pos, frame, shiftingRois).toRight("No ROIs for shifting!")
                        accuracy <- getRoisOutfile(pos, frame, accuracyRois).toRight("No ROIs for accuracy!")
                    } yield (shifting, accuracy)
                    maybeOutputPaths.leftMap((pos -> frame) -> _)
                })
                if (bads.nonEmpty) throw new Exception(s"${bads.length} (position, frame) pairs with problems.\n${bads}")
                val writeRois = (rois: NEL[IndexedRoi], outpath: os.Path) => {
                    println(s"Writing: $outpath")
                    val arr = IndexedRoi.toJsonSimple(parserConfig.coordinateSequence)(rois.toList)(using ((_: Coordinate).get).andThen(ujson.Num.apply))
                    os.makeDir.all(os.Path(outpath.toNIO.getParent))
                    os.write.over(outpath, ujson.write(arr, indent = 4))
                }
                goods foreach { case ((shiftingRois, shiftingPath), (accuracyRois, accuracyPath)) => 
                    writeRois(shiftingRois, shiftingPath)
                    writeRois(accuracyRois, accuracyPath)
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
        val numAvailable = rois.size
        (sampleSize <= numAvailable).either(s"Fewer ROIs available than requested! {numAvailable} < {sampleSize}", {
            val (inSample, outOfSample) = Random.shuffle(rois.toList).splitAt(sampleSize)
            val (forShifting, forAccuracy) = inSample.splitAt(numShifting)
            val shiftingRois = forShifting.map(roi => RoiForShifting(roi.index, roi.centroid))
            val accuracyRois = forAccuracy.map(roi => RoiForAccuracy(roi.index, roi.centroid))
            (outOfSample, shiftingRois, accuracyRois)
        })
    }

    /* Helper types */
    sealed trait ColumnName
    final case class XColumn(get: String) extends ColumnName
    given rwXcol: ReadWriter[XColumn] = readwriter[String].bimap(_.get, XColumn(_)) // Facilitate RW[ParserConfig] derivation.
    final case class YColumn(get: String) extends ColumnName
    given rwYcol: ReadWriter[YColumn] = readwriter[String].bimap(_.get, YColumn(_)) // Facilitate RW[ParserConfig] derivation.
    final case class ZColumn(get: String) extends ColumnName
    given rwZcol: ReadWriter[ZColumn] = readwriter[String].bimap(_.get, ZColumn(_)) // Facilitate RW[ParserConfig] derivation.
    
    final case class BadRecord(index: NonnegativeInt, record: RawRecord, problems: ErrorMessages)
    
    final case class ParserConfig(xCol: XColumn, yCol: YColumn, zCol: ZColumn, qcCol: String, coordinateSequence: CoordinateSequence) derives ReadWriter

    /* Helper functions */
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
