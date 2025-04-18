package at.ac.oeaw.imba.gerlich.looptrace

import scala.collection.immutable.SortedSet
import scala.util.{ NotGiven, Random, Try }
import cats.*
import cats.data.*
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*
import scopt.{ OParser, Read }
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.imaging.{
    FieldOfView,
    ImagingTimepoint, 
}
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.{
    FieldOfViewColumnName, 
    TimepointColumnName, 
}
import at.ac.oeaw.imba.gerlich.gerlib.json.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.json.syntax.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.nonnegativeInt.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Split pool of detected bead ROIs into those for drift correction shift, drift correction accuracy, and unused. */
object PartitionIndexedDriftCorrectionBeadRois extends ScoptCliReaders, StrictLogging:
    val ProgramName = "PartitionIndexedDriftCorrectionBeadRois"

    val BeadRoisPrefix = "bead_rois_"

    /* Type aliases */
    type RawRecord = Array[String]
    type FovTimePair = (FieldOfView, ImagingTimepoint)
    type KeyedProblem = (FovTimePair, RoisSplit.Problem)
    type InitFile = (FovTimePair, os.Path)
    type IndexedRoi = FiducialBead | SelectedRoi
    type JsonWriter[*] = upickle.default.Writer[*]

    final case class CliConfig(
        beadRoisRoot: os.Path = null, // bogus, unconditionally required
        numShifting: ShiftingCount = ShiftingCount(10), // bogus, unconditionally required
        numAccuracy: PositiveInt = PositiveInt(1), // bogus, unconditionally required
        outputFolder: Option[os.Path] = None, 
    )

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import parserBuilder.*
        given (intRead: Read[Int]) => Read[ShiftingCount] = 
            intRead.map(ShiftingCount.unsafe)

        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, BuildInfo.version), 
            opt[os.Path]("beadRoisRoot")
                .required()
                .action((p, c) => c.copy(beadRoisRoot = p))
                .validate(p => os.isDir(p).either(s"Alleged bead ROIs root isn't an extant folder: $p", ()))
                .text("Path to the folder with the detected bead ROIs"),
            opt[ShiftingCount]("numShifting")
                .required()
                .action((n, c) => c.copy(numShifting = n))
                .text("Number of ROIs to use for shifting"),
            opt[PositiveInt]("numAccuracy")
                .required()
                .action((n, c) => c.copy(numAccuracy = n))
                .text("Number of ROIs to use for accuracy"),
            opt[os.Path]('O', "outputFolder")
                .action((p, c) => c.copy(outputFolder = p.some))
                .text("Path to output root; if unspecified, use the input root."),
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => {
                workflow(
                    inputRoot = opts.beadRoisRoot, 
                    numShifting = opts.numShifting, 
                    numAccuracy = opts.numAccuracy, 
                    outputFolder = opts.outputFolder, 
                    )
            }
        }
    }

    /* Business logic */    
    def workflow(
        inputRoot: os.Path, 
        numShifting: ShiftingCount, 
        numAccuracy: PositiveInt, 
        outputFolder: Option[os.Path]
        ): Unit = {

        /* Actions */
        val inputFiles = discoverInputs(inputRoot)
        logger.info(s"Input file count: ${inputFiles.size}")
        val outfolder = outputFolder.getOrElse(inputRoot)
        logger.info(s"Will use output folder: $outfolder")
        
        // Write a specific subtype of selected ROI, but not a mix and not the general (non-leaf) type.
        def writeRois[R <: SelectedRoi : [R] =>> NotGiven[R =:= SelectedRoi]](rois: Set[R], outpath: os.Path): Unit = {
            logger.info(s"Writing: $outpath")
            val jsonObjs = rois.toList.map(SelectedRoi.toJsonSimple)
            os.makeDir.all(outpath.parent)
            os.write.over(outpath, ujson.write(jsonObjs, indent = 4))
        }
        /* Function definitions based on parsed config and CLI input */
        // Shifting ROIs cannot be empty.
        val writeRoisForShifting = (fovTime: FovTimePair, rois: NonEmptySet[RoiForShifting]) =>
            writeRois(rois.toSortedSet, getOutputFilepath(outfolder)(fovTime._1, fovTime._2, Purpose.Shifting))
        // Accuracy ROIs could be empty.
        val writeRoisForAccuracy = (fovTime: FovTimePair, rois: Set[RoiForAccuracy]) => 
            writeRois(rois, getOutputFilepath(outfolder)(fovTime._1, fovTime._2, Purpose.Accuracy))
        
        // Here, actually do the partition.
        val (bads, goods): (List[(InitFile, RoisSplit.Failure)], List[(InitFile, RoisSplit.HasPartition)]) = 
            Alternative[List].separate(
                preparePartitions(outfolder, numShifting = numShifting, numAccuracy = numAccuracy)(inputFiles.toList) map {
                    case (initFile, result) => (result match {
                        case p: RoisSplit.Partition => p.asRight      // ideal
                        case a: RoisSplit.TooFewAccuracy => a.asRight // OK (healthy or rescued)
                        case s: RoisSplit.TooFewShifting => s.asLeft  // bad
                        case e: RoisFileParseError => e.asLeft        // worst
                    }).bimap(initFile -> _, initFile -> _)
                }
            )
        
        // Here, write (severe) warnings (if in development mode) or raise an exception.
        val zeroAccuracyProblems = if bads.nonEmpty then {
            Alternative[List].separate(bads.map{ // Partition the list of problems by type of error.
                case kv@(_, _: RoisFileParseError) => kv.asLeft
                case ((pt, _), e: RoisSplit.TooFewShifting) => (pt, e).asRight
            }) match {
                case (Nil, tooFewErrors) => 
                    // In this case, there's at least one FOV in which there are too few ROIs to meet even the absolute minimum.
                    given writer: JsonWriter[KeyedProblem] = readWriterForKeyedTooFewProblem
                    val warningsFile = outfolder / "roi_partition_warnings.severe.json"
                    logger.warn(s"Writing severe warnings file: $warningsFile")
                    val (problemsToWrite, problemsToPropagate) = tooFewErrors.map{ 
                        (pt, tooFew) => (pt -> tooFew.shiftingProblem, pt -> tooFew.accuracyProblem) 
                    }.unzip
                    os.write(warningsFile, write(problemsToWrite, indent = 2))
                    problemsToPropagate
                case _ => 
                    // Here we have all parse errors or mixed error types and can't combine them.
                    throw new Exception(s"${bads.size} (FOV, timepoint) pairs with problems.\n${bads}")
            }
        } else { List.empty[KeyedProblem] }
        // NB: since possibly multiple problems per (pos, timepoint) pair (e.g., too few shifting and too few accuracy), 
        //     don't convert this to Map, since key collision is potentially problematic.
        val problems: List[KeyedProblem] = 
            zeroAccuracyProblems ::: goods.flatMap{ case ((pt, _), splitResult) => 
                /* Write the ROIs and emit the optional warning. */
                val partition = splitResult.partition
                writeRoisForShifting(pt, partition.shifting)
                writeRoisForAccuracy(pt, partition.accuracy)
                splitResult match {
                    case problematic: RoisSplit.Problematic => problematic.problems.toList.map(pt -> _)
                    case _ => List()
                }
            }.sortBy(_._1)(Order[FovTimePair].toOrdering)
        if (bads.isEmpty && problems.isEmpty) then logger.info("No warnings from bead ROIs partitioning, nice!")
        else {
            val warningsFile = outfolder / "roi_partition_warnings.json"
            logger.warn(s"Writing bead ROIs partition warnings file: $warningsFile")
            given writer: JsonWriter[KeyedProblem] = readWriterForKeyedTooFewProblem
            os.write(warningsFile, write(problems, indent = 2))
        }
        logger.info("Done!")
    }

    def createParser(header: RawRecord): ErrMsgsOr[RawRecord => ErrMsgsOr[FiducialBead]] = {
        import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.* // for >>> and >>, generally
        val maybeParseIndex = buildFieldParse(ParserConfig.indexCol, safeParseInt >>> RoiIndex.fromInt)(header)
        val maybeParseX = buildFieldParse(ParserConfig.xCol.get, safeParseDouble.andThen(_.map(XCoordinate.apply)))(header)
        val maybeParseY = buildFieldParse(ParserConfig.yCol.get, safeParseDouble.andThen(_.map(YCoordinate.apply)))(header)
        val maybeParseZ = buildFieldParse(ParserConfig.zCol.get, safeParseDouble.andThen(_.map(ZCoordinate.apply)))(header)
        // The QC flag parser maps empty String to true and nonempty String to false (nonempty indicates QC fail reasons.)
        (maybeParseIndex, maybeParseX, maybeParseY, maybeParseZ).tupled.toEither.map{
            case (parseIndex, parseX, parseY, parseZ) => { 
                (record: RawRecord) => (record.length === header.length)
                    .either(NonEmptyList.one(s"Header has ${header.length} fields but record has ${record.length}"), ())
                    .flatMap{ _ => 
                        val maybeIndex = parseIndex(record)
                        val maybeX = parseX(record)
                        val maybeY = parseY(record)
                        val maybeZ = parseZ(record)
                        (maybeIndex, maybeX, maybeY, maybeZ).mapN(
                            (i, x, y, z) => FiducialBead(i, Point3D(x, y, z))
                        ).toEither
                    }
            }
        }
    }

    def discoverInputs(inputsFolder: os.Path): Set[InitFile] = {
        val prepFileMeta: os.Path => Option[InitFile] = filepath => 
            BeadsFilenameDefinition.fromFilteredPath(filepath).map{ fnDef => (fnDef.fieldOfView -> fnDef.timepoint) -> filepath }
        val results = os.list(inputsFolder).filter(os.isFile).toList.flatMap(prepFileMeta)
        val histogram = results.groupBy(_._1).filter(_._2.length > 1)
        if histogram.nonEmpty then {
            given writeFiles: (Iterable[os.Path] => ujson.Value):
                def apply(paths: Iterable[os.Path]) = paths.map(_.last)
            val errMsg = s"Non-unique filenames for key(s): ${posTimeMapToJson("filepaths", histogram.view.mapValues(_.map(_._2)).toMap)}"
            throw new IllegalStateException(errMsg)
        }
        results.toSet
    }
    
    def preparePartitions(outputFolder: os.Path, numShifting: ShiftingCount, numAccuracy: PositiveInt): 
        List[InitFile] => List[(InitFile, RoisFileParseError | RoisSplit.Result)] = _.map { 
            case init@(_, roisFile) => init -> readRoisFile(roisFile).fold(
                identity, 
                sampleDetectedRois(numShifting = numShifting, numAccuracy = numAccuracy)
                )
        }

    /**
      * Read a single (one FOV, one time) ROIs file.
      * 
      * Potential "failures":
      * 1. Given path isn't a file
      * 2. Field delimiter can't be inferred from given path's extension
      * 3. Given file is empty
      * 4. One or more columns required (by {@code ParserConfig}) to parse aren't in file'e header (first line)
      * 5. Any record fails to parse
      * 
      * @param roisFile The file to parse
      * @return A collection of ROIs, representing what was detected for a particular (FOV, time) combo
      */
    def readRoisFile(roisFile: os.Path): Either[RoisFileParseError, Iterable[FiducialBead]] = {
        prepFileRead(roisFile)
            .toEither
            .flatMap{ case (sep, head, lines) => createParser(sep `split` head).map(_ -> lines.map(sep.split)) }
            .leftMap(RoisFileParseFailedSetup.apply)
            .flatMap { case (parse, rawRecords) => 
                Alternative[List].separate(
                    NonnegativeInt.indexed(rawRecords)
                        .map{ case (rr, i) => parse(rr).leftMap(errs => BadRecord(i, rr, errs)) }
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
      * @param rois Collection of detected bead ROIs, from a single (FOV, time) pair
      * @return An explanation of failure if partition isn't possible, or a partition with perhaps a warning
      */
    def sampleDetectedRois(numShifting: ShiftingCount, numAccuracy: PositiveInt)(rois: Iterable[FiducialBead]): RoisSplit.Result = 
        val sampleSize = numShifting + numAccuracy
        if sampleSize < numShifting || sampleSize < numAccuracy then {
            val msg = s"Appears overflow occurred computing sample size: ${numShifting} + ${numAccuracy} = ${sampleSize}"
            throw new IllegalArgumentException(msg)
        }
        val (inSample, _) = Random.shuffle(rois.toList) `splitAt` sampleSize
        val (shifting, remaining) = inSample `splitAt` numShifting
        val accuracy = remaining `take` numAccuracy
        RoisSplit.Partition.build(
            numShifting, 
            shifting.map(roi => RoiForShifting(roi.index, roi.centroid)), 
            numAccuracy, 
            accuracy.map(roi => RoiForAccuracy(roi.index, roi.centroid))
        )

    /******************************/
    /* Helper types and functions */
    /******************************/

    /**
      * Write a mapping, from FOV and time pair to value, to JSON.
      *
      * @param vKey The key to use for the {@code V} element in each object
      * @param timeByFov The mapping of data to write
      * @param writeV How to write each {@code V} element as JSON
      * @return A JSON array of object corresponding to each element of the map
      */
    def posTimeMapToJson[V](vKey: String, timeByFov: Map[(FieldOfView, ImagingTimepoint), V])(using writeV: (V) => ujson.Value): ujson.Value = {
        val proc1 = (pt: (FieldOfView, ImagingTimepoint), v: V) => ujson.Obj(
            FieldOfViewColumnName.value -> pt._1.asJson,
            TimepointColumnName.value -> pt._2.asJson,
            vKey -> writeV(v)
        )
        timeByFov.toList.map(proc1.tupled)
    }
    
    /** Write, to JSON, a pair of (FOV, image time) and a case of too-few-ROIs for shifting for drift correction. */
    private[looptrace] def readWriterForKeyedTooFewProblem: ReadWriter[KeyedProblem] = {
        import JsonMappable.*
        import FovTimePair.given
        import UJsonHelpers.UPickleCatsInstances.given
        readwriter[ujson.Value].bimap(
            (pair, problem) => JsonMappable
                .combineSafely(List(pair.toJsonMap, problem.toJsonMap))
                .fold(reps => throw RepeatedKeysException(reps), identity), 
            json => 
                val fovNel = Try{ FieldOfView.unsafeLift(json(FieldOfViewColumnName.value).int) }.toValidatedNel
                val timeNel = Try{ ImagingTimepoint.unsafe(json(FovTimePair.timeKey).int) }.toValidatedNel
                val reqdNel = Try{ PositiveInt.unsafe(json("requested").int) }.toValidatedNel
                val realNel = Try{ NonnegativeInt.unsafe(json("realized").int) }.toValidatedNel
                val purposeNel = Try{ read[Purpose](json("purpose")) }.toValidatedNel
                (fovNel, timeNel, reqdNel, realNel, purposeNel).mapN(
                    (fov, time, requested, realized, purpose) => 
                        val problem = purpose match {
                            case Purpose.Shifting => RoisSplit.Problem.shifting(requested, realized)
                            case Purpose.Accuracy => RoisSplit.Problem.accuracy(requested, realized)
                        }
                        (fov -> time) -> problem
                ) match {
                    case Validated.Invalid(errs) => 
                        val msg = f"${errs.size} error(s) reading pair of ((pos, time), too-few-ROIs): ${errs.map(_.getMessage)}"
                        throw new ujson.Value.InvalidData(json, msg)
                    case Validated.Valid(instance) => instance
                }
        )
    }

    final case class BeadsFilenameDefinition(fieldOfView: FieldOfView, timepoint: ImagingTimepoint):
        final def getUnfilteredInputFilename: String = getFilenameBase ++ ".csv"
        final def getFilteredInputFilename: String = getFilenameBase ++ ".filtered.csv"
        final def getOutputFilename(purpose: Purpose): String = getFilenameBase ++ "." ++ purpose.lowercase ++ ".json"
        private def getFilenameBase: String = s"${BeadRoisPrefix}_${fieldOfView.show_}_${timepoint.show_}"
    end BeadsFilenameDefinition

    object BeadsFilenameDefinition:        
        def fromFilteredPath = (_: os.Path).last match {
            case s"$prefix.filtered.csv" => fromPrefix(prefix)
            case _ => None
        }
        
        def fromUnfilteredPath = (_: os.Path).last match {
            case s"$prefix.csv" => fromPrefix(prefix)
            case _ => None
        }

        private def fromPrefix = (_: String) match {
            case s"bead_rois__${fov}_${time}" => fromRaw(fov, time)
            case _ => None
        }

        private def fromRaw(fov: String, time: String): Option[BeadsFilenameDefinition] = (
            readNonNegInt(fov).map(FieldOfView.apply), 
            readNonNegInt(time).map(ImagingTimepoint.apply)
        ).mapN(BeadsFilenameDefinition.apply)

        private def readNonNegInt: String => Option[NonnegativeInt] = s => 
            Try(s.toInt).toOption.flatMap(NonnegativeInt.maybe)
    end BeadsFilenameDefinition


    /** Refinement type for nonnegative integers */
    opaque type ShiftingCount <: Int = Int
    
    /** Helpers for working with nonnegative integers */
    object ShiftingCount:
        /** The absolute minimum number of bead ROIs required for drift correction */
        val AbsoluteMinimumShifting = ShiftingCount(10)

        inline def apply(z: Int): ShiftingCount = 
            inline if z < 10 then compiletime.error("Insufficient value (< 10) for shifting count!")
            else (z: ShiftingCount)
        extension (n: ShiftingCount)
            def asNonnegative: NonnegativeInt = NonnegativeInt.unsafe(n)
            def asPositive: PositiveInt = PositiveInt.unsafe(n)
        def unsafe(z: Int): ShiftingCount = 
            if z < AbsoluteMinimumShifting then 
                throw new IllegalArgumentException("Insufficient value (< 10) for shifting count!")
            else (z: ShiftingCount) 
    end ShiftingCount
    
    /** Tools for working with a fundamental grouping entity -- pair of FOV and imaging timepoint */
    object FovTimePair:
        private[PartitionIndexedDriftCorrectionBeadRois] val timeKey = "time"
        given jsonMappableForFovTimePair: JsonMappable[FovTimePair]:
            override def toJsonMap = (fov, time) => 
                Map(FieldOfViewColumnName.value -> fov.asJson, timeKey -> time.asJson)
    end FovTimePair

    /** Type for when something gois wrong parsing a bead ROIs file */
    sealed trait RoisFileParseError extends Throwable
    
    /** One or more error messages about what went wrong setting up the parse of a bead ROIs file. */
    final case class RoisFileParseFailedSetup(get: ErrorMessages) extends RoisFileParseError
    
    /** One or more non-parseable records from a bead ROIs file */
    final case class RoisFileParseFailedRecords(get: NonEmptyList[BadRecord]) extends RoisFileParseError
    
    object RoisSplit:
        type Failure = RoisFileParseError | TooFewShifting
        type RoiSplitOutcome = Either[Failure, HasPartition]
        type Result = TooFewShifting | HasPartition
        type TooFewAccuracy = TooFewAccuracyHealthy | TooFewAccuracyRescued
        type TooFew = TooFewShifting | TooFewAccuracy

        sealed trait HasPartition:
            def partition: Partition
        
        sealed trait Problematic:
            def problems: NonEmptyList[Problem]

        final case class Problem private(numRequested: PositiveInt, numRealized: NonnegativeInt, purpose: Purpose):
            /* Validation of reasonableness of arguments given that this is an alleged error / problem value being created */
            if numRealized > numRequested then 
                throw new IllegalArgumentException(s"Realized more ROIs than requested: $numRealized > $numRequested")
            if numRealized === numRequested.asNonnegative then
                throw new IllegalArgumentException(s"Alleged too few ROIs, but $numRealized = $numRequested")

        object Problem:
            import ShiftingCount.asPositive
            given jsonMappableForProblem: JsonMappable[Problem] = JsonMappable.instance{ 
                problem => Map(
                    "requested" -> ujson.Num(problem.numRequested), 
                    "realized" -> ujson.Num(problem.numRealized), 
                    "purpose" -> ujson.Str(problem.purpose.toString)
                )
            }
            def accuracy(numRequested: PositiveInt, numRealized: NonnegativeInt): Problem = 
                new Problem(numRequested, numRealized, Purpose.Accuracy)
            def shifting(numRequested: ShiftingCount, numRealized: NonnegativeInt): Problem = 
                new Problem(numRequested.asPositive, numRealized, Purpose.Shifting)
        end Problem

        final case class TooFewShifting(
            requestedShifting: ShiftingCount, 
            realizedShifting: NonnegativeInt, 
            requestedAccuracy: PositiveInt,
            ) extends Problematic:
            require(requestedShifting > realizedShifting, s"Alleged too few shifting ROIs, but $realizedShifting >= $requestedShifting")
            def shiftingProblem = Problem.shifting(requestedShifting, realizedShifting)
            def accuracyProblem = Problem.accuracy(requestedAccuracy, NonnegativeInt(0))
            override def problems: NonEmptyList[Problem] = NonEmptyList.of(shiftingProblem, accuracyProblem)
            def realizedAccuracy = NonnegativeInt(0)
        
        final case class TooFewAccuracyRescued(
            partition: Partition, 
            requestedShifting: ShiftingCount, 
            requestedAccuracy: PositiveInt,
            ) extends HasPartition with Problematic:
            require(partition.numShifting < requestedShifting, s"Alleged too few shifting ROIs, but ${partition.numShifting} >= $requestedShifting")
            require(
                partition.numAccuracy === NonnegativeInt(0), 
                s"Accuracy ROIs count should be 0 when there are too few shifting ROIs; got ${partition.numAccuracy}"
                )
            
            override def problems: NonEmptyList[Problem] = 
                import ShiftingCount.asNonnegative
                NonEmptyList.of(
                    Problem.shifting(requestedShifting, realizedShifting.asNonnegative), 
                    Problem.accuracy(requestedAccuracy, realizedAccuracy),
                )
            
            def realizedShifting = partition.numShifting
            
            def realizedAccuracy = partition.numAccuracy
        end TooFewAccuracyRescued

        final case class TooFewAccuracyHealthy(
            partition: Partition, 
            requestedAccuracy: PositiveInt,
            ) extends HasPartition with Problematic:
            require(
                partition.numAccuracy < requestedAccuracy, 
                s"Alleged too few accuracy ROIs but ${partition.numAccuracy} >= $requestedAccuracy"
                )
            override def problems: NonEmptyList[Problem] = 
                NonEmptyList.one(Problem.accuracy(requestedAccuracy, partition.numAccuracy))
            def realizedShifting = partition.numShifting
            def realizedAccuracy = partition.numAccuracy
        
        final case class Partition private(shifting: NonEmptySet[RoiForShifting], accuracy: Set[RoiForAccuracy]) extends HasPartition:
            import Point3D.given // for cats.Order
            given ev: Ordering[(RoiIndex, Point3D)] = Order[(RoiIndex, Point3D)].toOrdering
            require(
                (shifting.toSortedSet.map(roi => roi.index -> roi.centroid) & accuracy.map(roi => roi.index -> roi.centroid)).isEmpty, 
                s"ROIs for shifting overlap with ones for accuracy: ${(shifting.toSortedSet.map(roi => roi.index -> roi.centroid) & accuracy.map(roi => roi.index -> roi.centroid))}"
            )
            require(
                shifting.length >= ShiftingCount.AbsoluteMinimumShifting, 
                s"Not enough shifting ROIs: ${shifting.length} < ${ShiftingCount.AbsoluteMinimumShifting}"
            )
            final def partition = this
            final lazy val numShifting: ShiftingCount = ShiftingCount.unsafe(shifting.length)
            final lazy val numAccuracy = NonnegativeInt.unsafe(accuracy.size)
        end Partition

        object Partition:
            import at.ac.oeaw.imba.gerlich.looptrace.collections.*
            def build(reqShifting: ShiftingCount, shifting: List[RoiForShifting], reqAccuracy: PositiveInt, accuracy: List[RoiForAccuracy]): Result = {
                given orderingForSelectedRoi: Ordering[RoiForShifting] = orderSelectedRoisSimplified[RoiForShifting]
                (shifting.toNel, NonnegativeInt.unsafe(shifting.length)) match {
                    case (Some(shiftNel), numShifting) if numShifting >= ShiftingCount.AbsoluteMinimumShifting => 
                        val shiftingCounts = shiftNel.toList.groupBy(identity).view.mapValues(_.length).toMap
                        val accuracyCounts = accuracy.toList.groupBy(identity).view.mapValues(_.length).toMap
                        val shiftingRepeats = shiftingCounts.filter(_._2 > 1)
                        val accuracyRepeats = accuracyCounts.filter(_._2 > 1)
                        if shiftingRepeats.nonEmpty || accuracyRepeats.nonEmpty then {
                            throw RepeatedRoisWithinPartError(shiftingRepeats.toMap, accuracyRepeats.toMap)
                        }
                        val partition = new Partition(shiftingCounts.keySet.toNonEmptySetUnsafe, accuracyCounts.keySet)
                        if partition.numShifting < reqShifting then {
                            TooFewAccuracyRescued(partition, reqShifting, reqAccuracy)
                        } else if partition.numAccuracy < reqAccuracy then {
                            TooFewAccuracyHealthy(partition, reqAccuracy)
                        } else { partition }
                    case (_, numShifting) => RoisSplit.TooFewShifting(reqShifting, numShifting, reqAccuracy)
                }
            }

            case class RepeatedRoisWithinPartError private[Partition](shifting: Map[RoiForShifting, Int], accuracy: Map[RoiForAccuracy, Int]) extends Throwable:
                require(shifting.nonEmpty || accuracy.nonEmpty, s"Cannot build a ROI repeats exception with 2 empty maps!")
                require(shifting.forall(_._2 > 1), s"Cannot allege that a 'repeat' occurs fewer than two times: ${shifting.filter(_._2 < 2)}")
                require(accuracy.forall(_._2 > 1), s"Cannot allege that a 'repeat' occurs fewer than two times: ${accuracy.filter(_._2 < 2)}")
        end Partition

        final case class TooFewShiftingException(errors: NonEmptyList[(FovTimePair, TooFewShifting)]) 
            extends Exception(s"${errors.length} (FOV, time) pairs with insufficient ROIs for drift correction: $errors")
    end RoisSplit
    
    sealed trait ColumnName { def get: String }
    final case class XColumn(get: String) extends ColumnName
    final case class YColumn(get: String) extends ColumnName
    final case class ZColumn(get: String) extends ColumnName
    
    object ColumnName:
        given (liftStr: String => ujson.Value) => (ColumnName => ujson.Value) = cn => liftStr(cn.get)

    final case class BadRecord(index: NonnegativeInt, record: List[String], problems: ErrorMessages)
    object BadRecord:
        def apply(index: NonnegativeInt, record: RawRecord, problems: ErrorMessages) = new BadRecord(index, record.toList, problems)

    /** How the ROIs will be used w.r.t. drift correction (either actual shift, or accuracy assessment) */
    enum Purpose derives ReadWriter:
        case Shifting, Accuracy
        def lowercase = this.toString.toLowerCase
    
    /** Typeclass instances for working with bead purpose */
    object Purpose:
        given eqForPurpose: Eq[Purpose] = Eq.fromUniversalEquals[Purpose]
    
    /** Helpers for working with the parser configuration */
    object ParserConfig:
        val indexCol = "beadIndex"
        val xCol = XColumn("xc")
        val yCol = YColumn("yc")
        val zCol = ZColumn("zc")
    end ParserConfig

    /** Name ROIs subfolder according to how the selected ROIs are to be used. */
    def getOutputSubfolder(root: os.Path) = root / (_: Purpose).lowercase

    /** Name ROIs subfolder according to how the selected ROIs are to be used, and encode the purpose in the filename also. */
    def getOutputFilepath(root: os.Path)(pos: FieldOfView, time: ImagingTimepoint, purpose: Purpose): os.Path = 
        getOutputSubfolder(root)(purpose) / BeadsFilenameDefinition(pos, time).getOutputFilename(purpose)

    /** Infer delimiter and get header + data lines. */
    private def prepFileRead(roisFile: os.Path): ValidatedNel[String, (Delimiter, String, List[String])] = {
        val maybeSep = Delimiter.fromPath(roisFile).toRight(s"Cannot infer delimiter for file! $roisFile").toValidatedNel
        val maybeHeadTail = (os.read.lines(roisFile).toList match {
            case Nil => Left(f"No lines in file! $roisFile")
            case h :: t => Right(h -> t)
        }).toValidatedNel
        (maybeSep, maybeHeadTail).mapN{ case (sep, (h, t)) => (sep, h, t) }
    }

    private[looptrace] def orderSelectedRoisSimplified[R <: SelectedRoi]: Ordering[R] = 
        import Point3D.given // for cats.Order
        Order[(RoiIndex, Point3D)].contramap((roi: R) => roi.index -> roi.centroid).toOrdering

end PartitionIndexedDriftCorrectionBeadRois