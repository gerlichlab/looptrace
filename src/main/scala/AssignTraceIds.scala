package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.*
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import cats.syntax.all.*
import fs2.data.csv.*
import mouse.boolean.*
import scopt.*
import squants.space.{ Length, Nanometers }

import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.cell.NuclearDesignation
import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2
import at.ac.oeaw.imba.gerlich.gerlib.geometry.{ Centroid, DistanceThreshold, EuclideanDistance, ProximityComparable }
import at.ac.oeaw.imba.gerlich.gerlib.geometry.PiecewiseDistance.ConjunctiveThreshold
import at.ac.oeaw.imba.gerlich.gerlib.graph.{
    SimplestGraph,
    buildSimpleGraph,
}
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.{
    NucleusDesignationColumnName,
    SpotChannelColumnName,
}
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{
    readCsvToCaseClasses, 
    writeCaseClassesToCsv,
}
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.{
    RoiPartnersRequirementType,
    TraceIdDefinitionAndFiltrationRule,
}
import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.{
    MergeContributorsColumnNameForAssessedRecord,
    RoiIndexColumnName,
    TraceIdColumnName,
    TracePartnersColumName,
}
import at.ac.oeaw.imba.gerlich.looptrace.csv.getCsvRowDecoderForImagingChannel
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.IndexedDetectedSpot
import at.ac.oeaw.imba.gerlich.looptrace.space.{ BoundingBox, Pixels3D }

/** Assign trace IDs to regional spots, considering the potential to group some together for downstream analytical purposes. */
object AssignTraceIds extends ScoptCliReaders, StrictLogging:
    val ProgramName = "AssignTraceIds"

    final case class CliConfig(
        roundsConfig: ImagingRoundsConfiguration = null, // unconditionally required
        roisFile: os.Path = null, // unconditionally required
        pixels: Pixels3D = null, // required,
        outputFile: os.Path = null, // unconditionally required
    )

    val parserBuilder = OParser.builder[CliConfig]

    given Eq[os.Path] = Eq.fromUniversalEquals

    def main(args: Array[String]): Unit = {
        import parserBuilder.*

        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, BuildInfo.version), 
            opt[ImagingRoundsConfiguration]("configuration")
                .required()
                .action((rounds, cliConf) => cliConf.copy(roundsConfig = rounds))
                .text("Path to file specifying the imaging rounds configuration"),
            opt[os.Path]("roisFile")
                .required()
                .action((f, c) => c.copy(roisFile = f))
                .validate(f => os.isFile(f).either(s"Alleged ROIs file path isn't an extant file: $f", ()))
                .text("Path to the file with the ROIs for which to define trace IDs"),
            opt[Pixels3D]("pixels")
                .required()
                .action((ps, c) => c.copy(pixels = ps))
                .text("How many nanometers per unit in each direction (x, y, z)"),
            opt[os.Path]('O', "outputFile")
                .required()
                .action((f, c) => c.copy(outputFile = f)), 
            checkConfig{ c => 
                if c.roisFile =!= c.outputFile then success
                else failure(s"ROIs file and output file are the same! ${c.roisFile}")
            }
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help") // CLI parser gives error message.
            case Some(opts) => workflow(
                roundsConfig = opts.roundsConfig, 
                roisFile = opts.roisFile, 
                pixels = opts.pixels,
                outputFile = opts.outputFile,
            )
        }
    }

    private def checkTraceId(offLimits: NonEmptySet[TraceId])(tid: TraceId): Unit = 
        if (offLimits contains tid) {
            throw new Exception(s"Trace ID is already a ROI index and can't be used: ${tid.show_}")
        }

    private def definePairwiseDistanceThresholds(
        rules: NonEmptyList[TraceIdDefinitionAndFiltrationRule],
    ): Map[(ImagingTimepoint, ImagingTimepoint), DistanceThreshold] = 
        import AtLeast2.syntax.toList
        rules.map(_.mergeGroup)
            .map{ g =>
                g.members
                    .toList
                    .combinations(2)
                    .toList
                    .flatMap{
                        case t1 :: t2 :: Nil => 
                            val dt = g.distanceThreshold
                            List((t1 -> t2) -> dt, (t2 -> t1) -> dt)
                        case ts => 
                            throw new Exception(s"Got ${ts.length} elements when taking combinations of 2!")
                    }
            }
            .foldLeft(Map()){ (thresholds, g) => 
                g.foldLeft(thresholds){ case (acc, (k, v)) => 
                    if acc contains k 
                    then throw new Exception(s"Key $k is already mapped to a distance threshold!")
                    else acc + (k -> v)
                }
            }

    private def computeNeighborsGraph(
        rules: NonEmptyList[TraceIdDefinitionAndFiltrationRule],
        pixels: Pixels3D,
    )(records: NonEmptyList[InputRecord]): SimplestGraph[RoiIndex] = 
        val lookupProximity: Map[(ImagingTimepoint, ImagingTimepoint), ProximityComparable[InputRecord]] = 
            definePairwiseDistanceThresholds(rules)
                .view
                .mapValues{
                    case EuclideanDistance.Threshold(dt) => 
                        import at.ac.oeaw.imba.gerlich.looptrace.syntax.bifunctor.mapBoth
                        val thresholdSquared = scala.math.pow(dt.toDouble, 2)
                        new ProximityComparable[InputRecord]:
                            override def proximal: (InputRecord, InputRecord) => Boolean = (r1, r2) => 
                                val (p1, p2) = (r1, r2).mapBoth(_.centroid.asPoint)
                                val delX = pixels.liftX(p1.x.value - p2.x.value)
                                val delY = pixels.liftY(p1.y.value - p2.y.value)
                                val delZ = pixels.liftZ(p1.z.value - p2.z.value)
                                val distanceSquared = List(delX, delY, delZ).foldLeft(0.0){ 
                                    (sumSqs, pxDiff) => 
                                        val diff = (pxDiff in Nanometers).value
                                        sumSqs + scala.math.pow(diff, 2) 
                                }
                                distanceSquared < thresholdSquared
                    case ConjunctiveThreshold(_) => 
                        throw new Exception("For trace ID assignment, only Euclidean distance threshold is supported.")
                }
                .toMap
        val edgeEndpoints: Set[(RoiIndex, RoiIndex)] = 
            given Order[FieldOfViewLike] with
                override def compare(x: FieldOfViewLike, y: FieldOfViewLike): Int = (x, y) match {
                    case (fov1: FieldOfView, fov2: FieldOfView) => fov1 compare fov2
                    case (pos1: PositionName, pos2: PositionName) => pos1 compare pos2
                    case _ => throw new Exception(s"Cannot compare $x to $y")
                }
            records.groupBy(r => r.context.fieldOfView -> r.context.channel) // Only merge ROIs from the same context (FOV, channel).
                .values // Once records are properly grouped by context, we no longer care about those context keys.
                .flatMap(
                    // We do our pairwise calculations only within each group, but then flatten to collect all results.
                    _.toList.combinations(2).flatMap{  // flatMap here b/c of optionality of output from each record
                        case r1 :: r2 :: Nil =>
                            lookupProximity
                                // First, these records' timepoints may not have been in the rules set and 
                                // may therefore need to be tested for proximity.
                                .get(r1.timepoint -> r2.timepoint)
                                // Emit a pair of edge endpoints iff these records are proximal.
                                .flatMap(_.proximal(r1, r2).option(r1.index -> r2.index))
                        case notPair => 
                            throw new Exception(s"Got ${notPair.length} element(s) when taking pairs!")
                })
                .toSet
        // Ensure each record gets a node, and add the discovered edges.
        buildSimpleGraph(records.map(_.index).toList.toSet, edgeEndpoints)

    // Start trace IDs with 1 more than max ROI ID/index.
    private def getInitialTraceId(roiIds: NonEmptyList[RoiIndex]): TraceId = 
        val maxRoiId = roiIds.toList.max(using summon[Order[RoiIndex]].toOrdering)
        TraceId.unsafe(NonnegativeInt(1) + maxRoiId.get)

    private[looptrace] def labelRecordsWithTraceId(
        rules: NonEmptyList[TraceIdDefinitionAndFiltrationRule], 
        discardIfNotInGroupOfInterest: Boolean,
        pixels: Pixels3D,
    )(records: NonEmptyList[InputRecord]): NonEmptyList[OutputRecord] = 
        /* Necessary imports and type aliases */
        import AtLeast2.syntax.{ remove, toNes, toSet }
        type TimepointExpectationLookup = NonEmptyMap[ImagingTimepoint, TraceIdDefinitionAndFiltrationRule]
        
        val lookupRecord: NonEmptyMap[RoiIndex, InputRecord] = records.map(r => r.index -> r).toNem
        val lookupRule: TimepointExpectationLookup = 
            // Provide a way to get the expected group members and requirement stringency for a given timepoint.
            given orderForKeyValuePairs[V]: Order[(ImagingTimepoint, V)] = Order.by(_._1)
            given semigroup: Semigroup[TimepointExpectationLookup] = 
                Semigroup.instance{ (x, y) => 
                    val collisions = x.keys & y.keys
                    if collisions.isEmpty then x ++ y
                    else throw new Exception(s"${collisions.size} key collision(s) between lookups to combine: $collisions")
                }
            rules.reduceMap{ r => r.mergeGroup.members.toNes.map(_ -> r).toNonEmptyList.toNem }
        val initTraceId = getInitialTraceId(records.map(_.index))
        val traceIdsOffLimits = 
            // Don't use any ROI index/ID as a trace ID.
            records.map(_.index.get).map(TraceId.unsafe).toNes
        computeNeighborsGraph(rules, pixels)(records)
            .strongComponentTraverser()
            .map(_.nodes.map(_.outer) // Get ROI IDs.
                .toList.toNel // each component as a nonempty list
                .getOrElse{ throw new Exception("Empty component!") }) // protected against by definition of component
            .toList.toNel // We want a nonempty list of components to accumulate errors
            .getOrElse{ throw new Exception("No components!") } //  protected against by initial .toNel call on input ROIs
            .traverse(_.traverse{ i => lookupRecord.apply(i).toValidNel(i) })
            .fold(
                badIds => 
                    // guarded against by construction of the lookup from records input
                    throw new Exception(s"${badIds.length} ROI IDs couldn't be looked up! Here's one: ${badIds.head.show_}"),
                _.toList.foldRight(initTraceId -> List.empty[OutputRecord]){ 
                    case (recGroup, (currId, acc)) => 
                        checkTraceId(traceIdsOffLimits)(currId)
                        val newRecs: List[OutputRecord] = 
                            AtLeast2.either(recGroup.map(_.index).toList.toSet).fold(
                                Function.const{ // singleton
                                    given Eq[RoiPartnersRequirementType] = Eq.fromUniversalEquals
                                    val useRecord = lookupRule
                                        .apply(recGroup.head.timepoint)
                                        .fold(!discardIfNotInGroupOfInterest)(_.requirement === RoiPartnersRequirementType.Lackadaisical)
                                    if useRecord then List((recGroup.head, currId, None))
                                    else List()
                                }, 
                                multiIds => 
                                    val useGroup: Boolean = 
                                        recGroup // at least two ROIs in group/component
                                            .toList
                                            .flatMap{ r => lookupRule.apply(r.timepoint) }
                                            .toNel
                                            .fold(!discardIfNotInGroupOfInterest){ rules => 
                                                given Eq[TraceIdDefinitionAndFiltrationRule] = Eq.fromUniversalEquals
                                                val nUniqueRules = rules.toList.toSet.size
                                                if nUniqueRules =!= 1
                                                then throw new Exception(
                                                    s"$nUniqueRules unique merge rules (not 1) for single group!"
                                                )
                                                else
                                                    val rule = rules.head
                                                    val expectedTimes = rule.mergeGroup.members.toSet
                                                    val observedTimes = recGroup.map(_.timepoint).toList.toSet
                                                    rule.requirement match {
                                                        case RoiPartnersRequirementType.Conjunctive => 
                                                            observedTimes === expectedTimes
                                                        case RoiPartnersRequirementType.Disjunctive => 
                                                            observedTimes subsetOf expectedTimes
                                                        case RoiPartnersRequirementType.Lackadaisical => 
                                                            true
                                                    }
                                            }
                                    if useGroup 
                                    then recGroup.toList.map(rec => (rec, currId, multiIds.remove(rec.index).some))
                                    else List()
                            )
                        val newTid = TraceId.unsafe(NonnegativeInt(1) + currId.get)
                        (newTid, newRecs ::: acc)
                }
            )
            ._2
            .toNel
            .getOrElse{
                // guarded against by checking for empty input up front
                throw new Exception("Wound up with empty results despite nonempty input!")
            }

    def workflow(roundsConfig: ImagingRoundsConfiguration, roisFile: os.Path, pixels: Pixels3D, outputFile: os.Path): Unit = {
        import InputRecord.given
        import fs2.data.text.utf8.*
        given CsvRowDecoder[ImagingChannel, String] = 
            getCsvRowDecoderForImagingChannel(SpotChannelColumnName)
        
        IO{ logger.info(s"Reading ROIs file: $roisFile") }
            .flatMap{ Function.const(readCsvToCaseClasses[InputRecord](roisFile)) }
            .map(_.toNel match {
                case None => 
                    logger.error(s"No input record parsed from ROIs file ($roisFile)!")
                    Option.empty[NonEmptyList[OutputRecord]]
                case Some(records) => 
                    roundsConfig.mergeRules match {
                        case None => 
                            val initTraceId = getInitialTraceId(records.map(_.index))
                            val traceIdsOffLimits = records.map(r => TraceId(r.index.get)).toNes
                            records.zipWithIndex.map{ (r, i) => 
                                val newTid = TraceId.unsafe(NonnegativeInt.unsafe(i) + initTraceId.get)
                                checkTraceId(traceIdsOffLimits)(newTid)
                                (r, newTid, None)
                            }.some
                        case Some(rules) => 
                            labelRecordsWithTraceId(rules, roundsConfig.discardRoisNotInGroupsOfInterest, pixels)(records).some
                    }
            })
            .flatMap(_ match {
                case None => IO{ logger.error("No output to write!") }
                case Some(records) => 
                    import OutputRecord.given
                    given CsvRowEncoder[ImagingChannel, String] = 
                        // for derivation of CsvRowEncoder[ImagingContext, String]
                        SpotChannelColumnName.toNamedEncoder
                    logger.info(s"Writing output file: $outputFile")
                    fs2.Stream
                        .emits(records.sortBy(_._1.index).toList)
                        .through(writeCaseClassesToCsv[OutputRecord](outputFile))
                        .compile
                        .drain
            })
            .unsafeRunSync()
        
        logger.info("Done!")
    }

    private type OutputRecord = (InputRecord, TraceId, Option[NonEmptySet[RoiIndex]])

    object OutputRecord:
        given encOutRec(using 
            encRoiId: CellEncoder[RoiIndex],
            encContext: CsvRowEncoder[ImagingContext, String],
            encCentroid: CsvRowEncoder[Centroid[Double], String],
            encBox: CsvRowEncoder[BoundingBox, String], 
            encNuc: CellEncoder[NuclearDesignation],
            encTid: CellEncoder[TraceId],
        ): CsvRowEncoder[OutputRecord, String] with
            override def apply(elem: OutputRecord): RowF[Some, String] = 
                val (inrec, tid, maybePartners) = elem
                val idRow = RoiIndexColumnName.write(inrec.index)
                val ctxRow = encContext(inrec.context)
                val centerRow = encCentroid(inrec.centroid)
                val boxRow = encBox(inrec.box)
                val mergeInputsRow = 
                    MergeContributorsColumnNameForAssessedRecord.write(elem._1.maybeMergeInputs)
                val nucRow = 
                    elem._1.maybeNucleusDesignation match {
                        case None => RowF(
                            values = NonEmptyList.one(""), 
                            headers = Some(NonEmptyList.one(NucleusDesignationColumnName.value)),
                        )
                        case Some(nuclearDesignation) => 
                            NucleusDesignationColumnName.write(nuclearDesignation)
                    }
                val tidRow = TraceIdColumnName.write(tid)
                val tracePartnersRow = 
                    TracePartnersColumName.write(maybePartners.fold(Set())(_.toSortedSet.toSet))
                idRow |+| ctxRow |+| centerRow |+| boxRow |+| mergeInputsRow |+| nucRow |+| tidRow |+| tracePartnersRow
    end OutputRecord

    final case class InputRecord(
        index: RoiIndex, 
        context: ImagingContext, 
        centroid: Centroid[Double], 
        box: BoundingBox, 
        maybeMergeInputs: Set[RoiIndex],  // may be empty, as the input collection is possibly a mix of singletons and merge results
        maybeNucleusDesignation: Option[NuclearDesignation], // Allow the program to operate on non-nuclei-filtered ROIs.
    ):
        final def timepoint: ImagingTimepoint = context.timepoint

    /** Helpers for working with this program's input records */
    object InputRecord:
        given rowDecoderForInputRecord(using 
            decIndex: CellDecoder[RoiIndex],
            decContext: CsvRowDecoder[ImagingContext, String], 
            decCentroid: CsvRowDecoder[Centroid[Double], String],
            decBox: CsvRowDecoder[BoundingBox, String],
            decNuclus: CellDecoder[NuclearDesignation],
        ): CsvRowDecoder[InputRecord, String] = new:
            override def apply(row: RowF[Some, String]): DecoderResult[InputRecord] = 
                val spotNel = summon[CsvRowDecoder[IndexedDetectedSpot, String]](row)
                    .leftMap(e => s"Cannot decode spot from row ($row): ${e.getMessage}")
                    .toValidatedNel
                val mergeInputsNel = 
                    MergeContributorsColumnNameForAssessedRecord.from(row)
                val nucNel = 
                    val key = NucleusDesignationColumnName.value
                    row.apply(key) match {
                        // Allow the program to operate on non-nuclei-filtered ROIs.
                        case None | Some("") => Option.empty.validNel
                        case Some(s) => decNuclus(s)
                            .bimap(e => s"Cannot decode spot from row ($row): ${e.getMessage}", _.some)
                            .toValidatedNel
                    }
                (spotNel, mergeInputsNel, nucNel)
                    .mapN{ (spot, maybeMergeIndices, maybeNucleus) => 
                        InputRecord(spot.index, spot.context, spot.centroid, spot.box, maybeMergeIndices, maybeNucleus)
                    }
                    .toEither
                    .leftMap{ messages => 
                        DecoderError(s"${messages.length} error(s) reading row ($row):\n${messages.mkString_("\n")}")
                    }
    end InputRecord
end AssignTraceIds
