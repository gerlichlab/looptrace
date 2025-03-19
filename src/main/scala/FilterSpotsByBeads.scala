package at.ac.oeaw.imba.gerlich.looptrace

import java.nio.file.FileAlreadyExistsException
import scala.collection.immutable.SortedSet
import scala.collection.mutable.ListBuffer
import scala.util.Try
import cats.*
import cats.data.{ NonEmptyList, NonEmptySet }
import cats.effect.unsafe.implicits.global // for IORuntime
import cats.syntax.all.*
import fs2.data.csv.*
import fs2.data.text.utf8.byteStreamCharLike // for CharLikeChunks typeclass instances
import mouse.boolean.*
import scopt.*
import com.typesafe.scalalogging.StrictLogging
import at.ac.oeaw.imba.gerlich.gerlib.geometry.{ Centroid, DistanceThreshold, EuclideanDistance, Point3D, ProximityComparable }
import at.ac.oeaw.imba.gerlich.gerlib.imaging.{ FieldOfView, FieldOfViewLike, ImagingChannel, ImagingTimepoint, PositionName }
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.SpotChannelColumnName
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{ ColumnName, getCsvRowEncoderForProduct2, readCsvToCaseClasses, writeCaseClassesToCsv }
import at.ac.oeaw.imba.gerlich.gerlib.numeric.{ NonnegativeInt, NonnegativeReal, PositiveInt, PositiveReal }
import at.ac.oeaw.imba.gerlich.looptrace.PartitionIndexedDriftCorrectionBeadRois.{
    BeadRoisPrefix,
    BeadsFilenameDefinition, 
}
import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.drift.{ DriftRecord, Movement, TotalDrift }
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.csv.getCsvRowDecoderForImagingChannel
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.roi.AdmitsRoiIndex
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.IndexedDetectedSpot
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Filtration of FISH spots when they're too close to one or more fiducial beads */
object FilterSpotsByBeads extends StrictLogging, ScoptCliReaders:
    val ProgramName = "FilterSpotsByBeads"
    
    case class CliConfig(
        spotsFolder: os.Path = null, // required
        beadsFolder: os.Path = null, // required
        driftFile: os.Path = null, // required
        filteredOutputFile: os.Path = null, // required
        distanceThreshold: EuclideanDistance.Threshold = null, // required
        spotlessTimepoint: ImagingTimepoint = null, // required
        overwrite: Boolean = false,
    )

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = 
        import parserBuilder.*
        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, BuildInfo.version), 
            opt[os.Path]("spotsFolder")
                .required()
                .action((f, c) => c.copy(spotsFolder = f))
                .validate(f => os.isDir(f).either(s"Alleged spots file isn't extant folder: $f", ()))
                .text("Path to the folder with FISH spots to filter"), 
            opt[os.Path]("beadsFolder")
                .required()
                .action((f, c) => c.copy(beadsFolder = f))
                .validate(f => os.isDir(f).either(s"Alleged beads folder isn't extant folder: $f", ()))
                .text("Path to the folder in which to find bead ROI files"),
            opt[os.Path]("driftFile")
                .required()
                .action((f, c) => c.copy(driftFile = f))
                .validate(f => os.isFile(f).either(s"Alleged drift file isn't extant file: $f", ()))
                .text("Path to drift correction file"),
            opt[os.Path]("filteredOutputFile")
                .required()
                .action((f, c) => c.copy(filteredOutputFile = f))
                .text("Path to which to write the filtered spots (after discarding those proximal to a bead)"),
            opt[EuclideanDistance.Threshold]("distanceThreshold")
                .required()
                .action((dt, c) => c.copy(distanceThreshold = dt))
                .text("Definition of Euclidean distance beneath which a bead and a spot are considered proximal"),
            opt[ImagingTimepoint]("spotlessTimepoint")
                .required()
                .action((t, c) => c.copy(spotlessTimepoint = t))
                .text("Timepoint in which no FISH is done, such that all signal should be just beads (ideally)"),
            opt[Unit]("overwrite")
                .action((_, c) => c.copy(overwrite = true))
                .text("Authorise overwrite of any existing output file"),
            checkConfig(c => invalidateMainOutputState(c.overwrite, c.filteredOutputFile).fold(success)(failure))
        )

        OParser.parse(parser, args, CliConfig()) match
            case None => 
                // CLI parser gives error message.
                throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help")
            case Some(opts) => 
                import OneBasedFourDigitPositionName.toFieldOfView
                import ImagingChannelImplicits.given

                logger.info(s"Seeking spots files: ${opts.spotsFolder}")
                val spotsFiles = findSpotsFiles(opts.spotsFolder)
                logger.info(s"Spots file count: ${spotsFiles.length}")
                logger.info(s"Seeking beads files: ${opts.beadsFolder}")
                val lookupBeads = findBeadsFiles(opts.beadsFolder)
                    .foldLeft(Map.empty[FieldOfView, BeadsFilenameDefinition]){
                        (acc, beadsFnDef) => 
                            if beadsFnDef.timepoint === opts.spotlessTimepoint
                            then 
                                if acc contains beadsFnDef.fieldOfView
                                then throw new Exception("FOV already attributed to a beads file")
                                else acc + (beadsFnDef.fieldOfView -> beadsFnDef)
                            else acc
                    }
                logger.info(s"Beads file count: ${lookupBeads.size}")
                logger.info(s"Reading drifts: ${opts.driftFile}")
                val lookupTotalDrift = readCsvToCaseClasses[DriftRecord](opts.driftFile)
                    .unsafeRunSync()
                    .map(rec => 
                        val fovName: OneBasedFourDigitPositionName = forcePositionName(rec.fieldOfView)
                        (fovName -> rec.time) -> rec.total
                    )
                    .toMap
                
                val allGoodSpots: ListBuffer[IndexedDetectedSpot] = ListBuffer()
                Alternative[List].separate(
                    spotsFiles.map{ spotSpec => 
                        lookupBeads.get(spotSpec.fovName.toFieldOfView)
                            .toRight(spotSpec)
                            .map{ beadSpec => (spotSpec.fovName, spotSpec -> beadSpec) }
                    }
                ) match {
                    case (Nil, pairs) => pairs.sortBy(_._1).foreach{
                        case (fovName, (spotsFileDef, beadsFileDef)) => 
                            val beadDrift = lookupTotalDrift(fovName -> opts.spotlessTimepoint)
                            val spotsFile = opts.spotsFolder / spotsFileDef.getInputFilename
                            val beadsFile = opts.beadsFolder / beadsFileDef.getInputFilename
                            logger.info(s"Reading beads file: $beadsFile")
                            val beads = readCsvToCaseClasses[Bead](beadsFile)
                                .unsafeRunSync()
                                .foldLeft(Map.empty[RoiIndex, Point3D[Double]]){
                                    (acc, bead) => 
                                        val i = bead.index
                                        if acc contains i
                                        then throw new Exception(s"Repeated index in beads: $i")
                                        else acc + (i -> Movement.addDrift(beadDrift)(bead.centroid.asPoint))
                                }
                            
                            logger.info(s"Reading and processing spots from $spotsFile")
                            val (keeps, discards) = 
                                Alternative[List].separate(readCsvToCaseClasses[IndexedDetectedSpot](spotsFile)
                                    .unsafeRunSync()
                                    .map(spot => 
                                        val spotDrift = lookupTotalDrift(fovName -> spot.timepoint)
                                        val dcCenter = Movement.addDrift(spotDrift)(spot.centroid.asPoint)
                                        val nearBeads = 
                                            findNeighbors[Double, Point3D[Double], Point3D[Double]](beads, opts.distanceThreshold)(dcCenter)
                                        NonEmptySet.fromSet(SortedSet.from(nearBeads)(Order[RoiIndex].toOrdering))
                                            .toRight(spot)
                                            .map(spot -> _)
                                    )
                            )
                            // Add the keepers to the collection of spots.
                            allGoodSpots ++= keeps
                            /* Write the discarded records. */
                            val discardsFile: os.Path = spotsFile.parent / (fovName ++ "__discarded_for_beads.csv")
                            if !opts.overwrite && os.exists(discardsFile) then {
                                throw new FileAlreadyExistsException(f"Discards file already exists: $discardsFile")
                            }
                            fs2.Stream.emits(discards)
                                .map(Discard.fromSpotAndBeads.tupled)
                                .through(writeCaseClassesToCsv(discardsFile))
                                .compile
                                .drain
                                .unsafeRunSync()
                    }
                    case (bads, _) => 
                        throw new Exception(s"${bads.length} spots specs without beads: $bads")
                }

                invalidateMainOutputState(opts.overwrite, opts.filteredOutputFile)
                    .foreach(msg => throw new FileAlreadyExistsException(msg))
                logger.info(s"Writing filtered output: ${opts.filteredOutputFile}")
                fs2.Stream.emits(allGoodSpots)
                    .through(writeCaseClassesToCsv[IndexedDetectedSpot](opts.filteredOutputFile))
                    .compile
                    .drain
                    .unsafeRunSync()

                logger.info("Done!")
                
    final case class Discard private(spotIndex: RoiIndex, beadIndices: NonEmptySet[RoiIndex])

    private def invalidateMainOutputState(overwrite: Boolean, target: os.Path): Option[String] = 
        (!overwrite && os.exists(target)).option(s"Target already exists: $target")

    object Discard:
        def fromSpotAndBeads(spot: IndexedDetectedSpot, beads: NonEmptySet[RoiIndex]) = 
            new Discard(spot.index, beads)

        given CsvRowEncoder[Discard, String] = 
            given CellEncoder[NonEmptySet[RoiIndex]] = summon[CellEncoder[Set[RoiIndex]]].contramap(_.toSortedSet)
            new:
                override def apply(elem: Discard): RowF[Some, String] = 
                    val spotText = ColumnName[RoiIndex]("spotIndex").write(elem.spotIndex)
                    val beadsText = ColumnName[NonEmptySet[RoiIndex]]("beadIndices").write(elem.beadIndices)
                    spotText |+| beadsText
    end Discard

    def forcePositionName(fovLike: FieldOfViewLike): OneBasedFourDigitPositionName =
        val maybePosName = fovLike match {
            case fov: FieldOfView => OneBasedFourDigitPositionName.fromFieldOfView(fov)
            case pos: PositionName => OneBasedFourDigitPositionName.fromPositionName(pos)
                        
        }
        maybePosName.fold(
            msg => throw new IllegalArgumentException(
                s"Cannot refine given value ($fovLike) to one-based, four-digit name: $msg"
            ),
            identity
        )

    final case class Bead(index: RoiIndex, centroid: Centroid[Double])

    object Bead:
        given (
            decIdx: CellDecoder[RoiIndex], 
            decCenter: CsvRowDecoder[Centroid[Double], String],
        ) => CsvRowDecoder[Bead, String] = 
            new:
                override def apply(row: RowF[Some, String]): DecoderResult[Bead] = 
                    val indexNel = ColumnName[RoiIndex]("beadIndex").from(row)
                    val centerNel = decCenter(row)
                        .leftMap(err => NonEmptyList.one(err.getMessage))
                        .toValidated
                    (indexNel, centerNel)
                        .mapN(Bead.apply)
                        .toEither
                        .leftMap(messages => 
                            DecoderError(s"Problem(s) decoding bead: ${messages.mkString_("; ")}")
                        )
    end Bead

    final case class SpotsFileDefinition(fovName: OneBasedFourDigitPositionName):
        def getInputFilename: String = fovName ++ "_rois.csv"

    object SpotsFileDefinition:
        import scala.util.chaining.*
        
        def fromFilepath = (_: os.Path).last.pipe(fromFilename)
        
        private def fromFilename = (_: String) match {
            case s"${rawPosName}_rois.csv" => 
                OneBasedFourDigitPositionName
                    .fromString(false)(rawPosName)
                    .map(SpotsFileDefinition.apply)
                    .toOption
            case _ => None
        }
    end SpotsFileDefinition

    def findSpotsFiles(spotsFolder: os.Path): List[SpotsFileDefinition] = 
        os.list(spotsFolder).toList.flatMap(SpotsFileDefinition.fromFilepath)

    def findBeadsFiles(beadsFolder: os.Path): List[BeadsFilenameDefinition] = 
        require(os.isDir(beadsFolder), s"Alleged beads folder isn't an extant folder: $beadsFolder")
        os.list(beadsFolder).toList.flatMap(BeadsFilenameDefinition.fromPath)

    def findNeighbors[C: Numeric, Ref, Query](
        refs: Map[RoiIndex, Point3D[C]], 
        threshold: DistanceThreshold
    )(using CenterFinder[Query, C]) = 
        import CenterFinder.syntax.*
        import ProximityComparable.proximal
        given ProximityComparable[Point3D[C]] = DistanceThreshold.defineProximityPointwise(threshold)
        (query: Query) => 
            refs.foldLeft(Set.empty[RoiIndex]){ 
                case (acc, (i, p)) => if query.locateCenter.asPoint `proximal` p then acc + i else acc 
            }

    def findNeighbors[C: Numeric, Ref, Query](
        refs: Map[RoiIndex, Ref], 
        threshold: DistanceThreshold
    )(using CenterFinder[Ref, C], CenterFinder[Query, C]): Query => Set[RoiIndex] = 
        import CenterFinder.syntax.*
        findNeighbors(refs.map{ (i, r) => i -> r.locateCenter.asPoint }, threshold)

    trait CenterFinder[A, C]:
        def locate(a: A): Centroid[C]

    object CenterFinder:
        def instance[A, C]: (A => Centroid[C]) => CenterFinder[A, C] = 
            f => new:
                override def locate(a: A): Centroid[C] = f(a)
        
        given [C] => CenterFinder[Point3D[C], C] = instance(Centroid.fromPoint)

        given [C] => CenterFinder[Centroid[C], C] = instance(identity)

        given [C] => (Contravariant[[a] =>> CenterFinder[a, C]]) = new:
            override def contramap[A, B](fa: CenterFinder[A, C])(f: B => A): CenterFinder[B, C] = 
                new:
                    override def locate(b: B): Centroid[C] = fa.locate(f(b))

        object syntax:
            extension [A](a: A)
                def locateCenter[C](using finder: CenterFinder[A, C]): Centroid[C] = 
                    finder.locate(a)
    end CenterFinder

    /**
      * Given instances related to [[at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingChannel]]
      * 
      * These are necessary in certain spots for the derivation of a given/implicit 
      * instance for a type of which an imaging channel is a component.
      */
    object ImagingChannelImplicits:
        given CsvRowDecoder[ImagingChannel, String] = 
            getCsvRowDecoderForImagingChannel(SpotChannelColumnName)
        
        given (CellEncoder[ImagingChannel]) => CsvRowEncoder[ImagingChannel, String] = 
            SpotChannelColumnName.toNamedEncoder
    end ImagingChannelImplicits
end FilterSpotsByBeads
