package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ NotGiven, Try }
import upickle.default.*
import cats.*
import cats.data.*
import cats.effect.IO
import cats.effect.unsafe.IORuntime
import cats.effect.unsafe.implicits.global // needed for cats.effect.IORuntime
import cats.syntax.all.*
import fs2.Stream
import fs2.data.csv.*
import fs2.data.text.utf8.byteStreamCharLike
import mouse.boolean.*

import scopt.OParser
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.geometry.{
    AxisX, 
    AxisY, 
    AxisZ, 
    BoundingBox as BBox, 
    Centroid, 
    DistanceThreshold, 
    PiecewiseDistance, 
    ProximityComparable,
}
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{
    ColumnName, 
    getCsvRowDecoderForSingleton,
    getCsvRowEncoderForSingleton,
    readCsvToCaseClasses, 
    writeCaseClassesToCsv,
}
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.{
    SpotChannelColumnName,
    TimepointColumnName
}
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.nonnegativeInt.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.syntax.all.*
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.collections.*
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.{
    MergeRoisColumnName,
    RoiIndexColumnName,
}
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.drift.*
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.roi.{
    AdmitsRoiIndex,
    DetectedSpotRoi,
    MergedRoiRecord,
}
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.{
    IndexedDetectedSpot, 
    PostMergeRoi,
}
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.IndexedDetectedSpot.given
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.PostMergeRoi.*
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.PostMergeRoi.given
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.roi.UnidentifiableRoi
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.assessForMutualExclusion
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.NamedRow

/**
 * Filter out spots that are too close together (e.g., because disambiguation 
 * of spots from indiviudal FISH probes in each region would be impossible in a multiplexed 
 * experiment).
 * 
 * If doing proximity-based filtration of spots, the value for the `"semantic"` key in the config 
 * file in the `"proximityFilterStrategy"` section should be either `"permissive"` or `"prohibitive"`. 
 * To treat all regional spots as a single group and prohibit proximal spots, the `"proximityFilterStrategy"` 
 * section may be omitted. To do 'no' proximity-based filtration, the minimum spot distance \
 * could be set to 0. 
 * 
 * For prohibitions, the groupings are interpreted by the program to mean that spots from the 
 * grouped regional barcodes may 'not' violate the proximity threshold; spots from other pairs 
 * of regional barcodes 'are' permitted to violate that threshold. 
 * For permissions, spots from pairs of regional barcodes in the same group 'may' violate 
 * the threshold while all others may 'not'.
 * 
 * For more, see related discussions on Github:
 * [[https://github.com/gerlichlab/looptrace/issues/71 Original, prohibitive semantics]]
 * [[https://github.com/gerlichlab/looptrace/issues/198 Newer, permissive semantics]]
 * [[https://github.com/gerlichlab/looptrace/issues/215 How to declare imaging rounds]]
 * [[https://github.com/gerlichlab/looptrace/pull/267 First config redesign]]
 * 
 * @author Vince Reuter
 */
object FilterRoisByProximity extends ScoptCliReaders, StrictLogging:
    val ProgramName = "FilterRoisByProximity"

    /**
      * The command-line configuration/interface definition
      *
      * @param configuration The configuration of the imaging rounds
      * @param spotsFile Path to the regional spots file in which to label records as too proximal or not
      * @param driftFile Path to the file with drift correction information for all ROIs, across all timepoints
      * @param fileForDiscards Path to which to write discarded ROIs (too close together)
      * @param fileForKeepers Path to which to write kept ROIs (not too close together)
      * @param extantOutputHandler How to handle if an output target already exists
      */
    final case class CliConfig(
        configuration: ImagingRoundsConfiguration = null, // unconditionally required
        spotsFile: os.Path = null, // unconditionally required
        driftFile: os.Path = null, // unconditionally required
        fileForDiscards: os.Path = null, // unconditionally required
        fileForKeepers: os.Path = null, // unconditionally required
        overwrite: Boolean = false,
        )

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
            opt[os.Path]("spotsFile")
                .required()
                .action((f, c) => c.copy(spotsFile = f))
                .validate(f => os.isFile(f).either(f"Alleged spots file isn't a file: $f", ()))
                .text("Path to regional spots file"),
            opt[os.Path]("driftFile")
                .required()
                .action((f, c) => c.copy(driftFile = f))
                .validate(f => os.isFile(f).either(f"Alleged drift file isn't a file: $f", ()))
                .text("Path to drift correction file"),
            opt[os.Path]("fileForDiscards")
                .required()
                .action((f, c) => c.copy(fileForDiscards = f))
                .text("Path to file for ROIs to discard on account of being too close together"),
            opt[os.Path]("fileForKeepers")
                .required()
                .action((f, c) => c.copy(fileForKeepers = f))
                .text("Path to file to which to write ROIs to continue processing, based on being sufficiently well separated"),
            opt[Unit]("overwrite")
                .action((_, c) => c.copy(overwrite = true))
                .text("License overwriting existing output"),
            checkConfig{ c => 
                val filepaths = List(c.spotsFile, c.driftFile, c.fileForDiscards, c.fileForKeepers)
                if filepaths.length === filepaths.toSet.size then success
                else failure(s"Repeat(s) are present among given filepaths: ${filepaths.mkString(", ")}")
            }, 
            checkConfig{ c => 
                if c.overwrite then success
                else if os.exists(c.fileForDiscards) then failure(s"Discards file exists: ${c.fileForDiscards}")
                else if os.exists(c.fileForKeepers) then failure(s"Keepers file exists: ${c.fileForKeepers}")
                else success
            }
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => workflow(
                spotsFile = opts.spotsFile, 
                driftFile = opts.driftFile, 
                proximityFilterStrategy = opts.configuration.proximityFilterStrategy, 
                fileForDiscards = opts.fileForDiscards,
                fileForKeepers = opts.fileForKeepers, 
                overwrite = opts.overwrite,
            )
        }
    }

    def workflow(
        spotsFile: os.Path, 
        driftFile: os.Path, 
        proximityFilterStrategy: ImagingRoundsConfiguration.ProximityFilterStrategy,
        fileForDiscards: os.Path, 
        fileForKeepers: os.Path, 
        overwrite: Boolean,
        ): Unit = {
        require(
            fileForDiscards.toIO.getPath =!= fileForKeepers.toIO.getPath, 
            s"Discards and keepers output filepaths match: ${(fileForDiscards, fileForKeepers)}"
        )
        
        // The first program in the chain: read in the ROIs, a possible mix  of singletons and merged records.
        val readRois: IO[List[PostMergeRoi]] = 
            readCsvToCaseClasses[PostMergeRoi](spotsFile)

        val keyDrifts: List[DriftRecord] => Either[String, Map[DriftKey, DriftRecord]] = drifts => 
            val (recordNumbersByKey, keyed) = 
                NonnegativeInt.indexed(drifts)
                    .foldLeft(Map.empty[DriftKey, NonEmptySet[NonnegativeInt]] -> Map.empty[DriftKey, DriftRecord]){ 
                        case ((reps, acc), (drift, recnum)) =>  
                            val fov = drift.fieldOfView
                            val t = drift.time
                            val k = fov -> t
                            reps.get(k) match {
                                case None => (reps + (k -> NonEmptySet.one(recnum)), acc + (k -> drift))
                                case Some(prevLineNums) => (reps + (k -> prevLineNums.add(recnum)), acc)
                            }
                    }
            val repeats = recordNumbersByKey.filter(_._2.size > 1)
            if repeats.isEmpty then keyed.asRight
            else { 
                val simpleReps = repeats.toList.map{ (k, lineNums) => k -> lineNums.toList.sorted }
                s"${simpleReps.length} repeated (FOV, time) pairs: ${simpleReps}".asLeft
            }

        // TODO: need to bring in the CsvRowDecoder[DriftRecord, String] instance
        val readKeyedDrifts: IO[Either[String, Map[DriftKey, DriftRecord]]] = 
            given CsvRowDecoder[FieldOfViewLike, String] = 
                getCsvRowDecoderForSingleton(ColumnName[FieldOfViewLike]("fieldOfView"))
            given CsvRowDecoder[ImagingTimepoint, String] = 
                getCsvRowDecoderForSingleton(TimepointColumnName)
            readCsvToCaseClasses[DriftRecord](driftFile).map(keyDrifts)

        given Eq[FieldOfViewLike] with
            override def eqv(a: FieldOfViewLike, b: FieldOfViewLike): Boolean = (a, b) match {
                case (fov1: PositionName, fov2: PositionName) => fov1 === fov2
                case (fov1: FieldOfView, fov2: FieldOfView) => fov1 === fov2
                case (_: PositionName, _: FieldOfView) => false
                case (_: FieldOfView, _: PositionName) => false
            }

        val writeOutputs: (List[UnidentifiableRoi], List[PostMergeRoi]) => IO[(os.Path, os.Path)] = 
            (unidentifiables, wellSeparatedRois) => 
                given CsvRowEncoder[ImagingChannel, String] = 
                    getCsvRowEncoderForSingleton(SpotChannelColumnName)
                val writeDiscards = Stream.emits(unidentifiables)
                    .through(writeCaseClassesToCsv(fileForDiscards))
                    .compile
                    .drain
                val writeKeepers = Stream.emits(wellSeparatedRois)
                    .through(writeCaseClassesToCsv(fileForKeepers))
                    .compile
                    .drain
                for {
                    _ <- writeDiscards
                    _ <- writeKeepers
                } yield (fileForDiscards, fileForKeepers)

        val program: IO[Unit] = for {
            rois <- readRois
            maybeDrifts <- readKeyedDrifts.map(_.leftMap(msg => new Exception(msg)))
            shiftedRois = maybeDrifts.flatMap(keyedDrifts => applyDrifts(keyedDrifts)(rois)).leftMap(NonEmptyList.one)
            (unidentifiablesFile, wellSeparatedsFile) <- 
                shiftedRois.map(assessForMutualExclusion(proximityFilterStrategy)) match {
                    case Left(errors) => 
                        throw new Exception(s"${errors.length} error(s). First one: ${errors.head}")
                    case Right((unidentifiables, postMergers)) => 
                        writeOutputs(unidentifiables, postMergers)
            }
            _ <- IO{ logger.info(s"Wrote unidentifiable ROIs to file: ${unidentifiablesFile}") }
            _ <- IO{ logger.info(s"Wrote well-separated ROIs to file: ${wellSeparatedsFile}") }
        } yield ()

        program.unsafeRunSync()

        logger.info("Done!")
    }

    private def applyDrifts(keyedDrifts: Map[DriftKey, DriftRecord])(using Eq[FieldOfViewLike]): List[PostMergeRoi] => Either[Throwable, List[PostMergeRoi]] = 
        val tryApp: PostMergeRoi => Either[Throwable, PostMergeRoi] = r => 
            val fovTimePair = r.context.fieldOfView -> r.context.timepoint
            keyedDrifts.get(fovTimePair)
                .toRight(DriftRecordNotFoundError(fovTimePair))
                .flatMap{ drift => Try{ applyDrift(r, drift) }.toEither }
        _.traverse(tryApp)

    /****************************************************************************************************************
     * Main types and business logic
     ****************************************************************************************************************/
    final case class DriftRecordNotFoundError(key: DriftKey) extends NoSuchElementException(s"key not found: ($key)")

    /** Add the given total drift (coarse + fine) to the given ROI, updating its centroid and its bounding box accordingly. */
    private def applyDrift(roi: PostMergeRoi, drift: DriftRecord)(using Eq[FieldOfViewLike]): PostMergeRoi = {
        require(
            roi.context.fieldOfView === drift.fieldOfView && roi.context.timepoint === drift.time, 
            s"ROI and drift don't match on (FOV, time): (${roi.context.fieldOfView -> roi.context.timepoint} and (${drift.fieldOfView -> drift.time})"
            )
        val (center, box) = PostMergeRoi.getCenterAndBox(roi)
        val newCenter = Centroid.fromPoint(Movement.addDrift(drift.total)(center.asPoint))
        val newBox = Movement.addDrift(drift.total)(box)
        roi match {
            case (idx, DetectedSpotRoi(spot, _)) => idx -> DetectedSpotRoi(spot.copy(centroid = newCenter), newBox)
            case rec: MergedRoiRecord => rec.copy(centroid = newCenter, box = newBox)
        }
    }

    /****************************************************************************************************************
     * Ancillary definitions
     ****************************************************************************************************************/
    object Movement:
        private type Interval[C <: Coordinate] = BBox.Interval[Double, C]
        def shiftBy(del: TotalDriftComponent[AxisX])(c: XCoordinate): XCoordinate = XCoordinate(del.value) |+| c
        def shiftBy(del: TotalDriftComponent[AxisY])(c: YCoordinate): YCoordinate = YCoordinate(del.value) |+| c
        def shiftBy(del: TotalDriftComponent[AxisZ])(c: ZCoordinate): ZCoordinate = ZCoordinate(del.value) |+| c
        def addDrift(drift: TotalDrift)(pt: Point3D): Point3D = 
            Point3D(shiftBy(drift.x)(pt.x), shiftBy(drift.y)(pt.y), shiftBy(drift.z)(pt.z))
        def addDrift(drift: TotalDrift)(box: BoundingBox): BoundingBox = BBox(
            BBox.Interval(shiftBy(drift.x)(box.sideX.lo), shiftBy(drift.x)(box.sideX.hi)),
            BBox.Interval(shiftBy(drift.y)(box.sideY.lo), shiftBy(drift.y)(box.sideY.hi)), 
            BBox.Interval(shiftBy(drift.z)(box.sideZ.lo), shiftBy(drift.z)(box.sideZ.hi)),
        )
    end Movement

    /* Type aliases */
    type DriftKey = (FieldOfViewLike, ImagingTimepoint)
end FilterRoisByProximity
