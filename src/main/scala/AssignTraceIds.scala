package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.*
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import cats.syntax.all.*
import fs2.data.csv.*
import mouse.boolean.*
import scopt.*

import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.cell.NucleusNumber
import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2
import at.ac.oeaw.imba.gerlich.gerlib.geometry.{ Centroid, DistanceThreshold, ProximityComparable }
import at.ac.oeaw.imba.gerlich.gerlib.graph.{
    SimplestGraph,
    buildSimpleGraph,
}
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.SpotChannelColumnName
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.readCsvToCaseClasses
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
}
import at.ac.oeaw.imba.gerlich.looptrace.csv.getCsvRowDecoderForImagingChannel
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.IndexedDetectedSpot
import at.ac.oeaw.imba.gerlich.looptrace.space.BoundingBox
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.ProximityGroup

/** Assign trace IDs to regional spots, considering the potential to group some together for downstream analytical purposes. */
object AssignTraceIds extends ScoptCliReaders, StrictLogging:
    val ProgramName = "AssignTraceIds"

    final case class CliConfig(
        roundsConfig: ImagingRoundsConfiguration = null, // unconditionally required
        roisFile: os.Path = null, // unconditionally required
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
                outputFile = opts.outputFile,
            )
        }
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

    private def computeNeighborsGraph(rules: NonEmptyList[TraceIdDefinitionAndFiltrationRule])(records: NonEmptyList[InputRecord]): SimplestGraph[RoiIndex] = 
        val lookupProximity: Map[(ImagingTimepoint, ImagingTimepoint), ProximityComparable[InputRecord]] = 
            definePairwiseDistanceThresholds(rules)
                .view
                .mapValues{ dt => DistanceThreshold.defineProximityPointwise(dt)((_: InputRecord).centroid.asPoint) }
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

    private def labelRecords(
        rules: NonEmptyList[TraceIdDefinitionAndFiltrationRule], 
        discardIfNotInGroupOfInterest: Boolean,
    )(maybeRecords: List[InputRecord]): Option[NonEmptyList[(InputRecord, TraceId, Option[NonEmptySet[RoiIndex]])]] = 
        /* Necessary imports and type aliases */
        import AtLeast2.syntax.{ remove, toNes, toSet }
        import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given // SimpleShow instances for domain types
        type TimepointExpectationLookup = NonEmptyMap[ImagingTimepoint, TraceIdDefinitionAndFiltrationRule]
        
        maybeRecords.toNel.map{ records => 
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
            val initTraceId = 
                // Start trace IDs with 1 more than max ROI ID/index.
                TraceId.unsafe(NonnegativeInt(1) + records.foldLeft(records.head.index){ (i, r) => i max r.index }.get)
            val traceIdsOffLimits = 
                // Don't use any ROI index/ID as a trace ID.
                records.map(_.index.get).map(TraceId.unsafe).toNes
            computeNeighborsGraph(rules)(records)
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
                    _.toList.foldRight(initTraceId -> List.empty[(InputRecord, TraceId, Option[NonEmptySet[RoiIndex]])]){ 
                        case (recGroup, (currId, acc)) => 
                            if (traceIdsOffLimits contains currId) {
                                // guarded against by starting with max ROI index and always incrementing the currId
                                throw new Exception(s"Trace ID is already a ROI index and can't be used: ${currId.show_}")
                            }
                            val newRecs: List[(InputRecord, TraceId, Option[NonEmptySet[RoiIndex]])] = 
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
        }

    def workflow(roundsConfig: ImagingRoundsConfiguration, roisFile: os.Path, outputFile: os.Path): Unit = {
        val readRois: IO[List[InputRecord]] = 
            import InputRecord.given
            import fs2.data.text.utf8.*
            given CsvRowDecoder[ImagingChannel, String] = 
                getCsvRowDecoderForImagingChannel(SpotChannelColumnName)
            readCsvToCaseClasses(roisFile)
        logger.info(s"Reading ROIs file: $roisFile")
        logger.info("Done!")
    }

    final case class InputRecord(
        index: RoiIndex, 
        context: ImagingContext, 
        centroid: Centroid[Double], 
        box: BoundingBox, 
        maybeMergeInputs: Set[RoiIndex],  // may be empty, as the input collection is possibly a mix of singletons and merge results
        maybeNucleusNumber: Option[NucleusNumber], // allow the program to operate on non-nuclei-filtered ROIs.
    ):
        final def timepoint: ImagingTimepoint = context.timepoint

    /** Helpers for working with this program's input records */
    object InputRecord:
        given rowDecoderForInputRecord(using 
            decIndex: CellDecoder[RoiIndex],
            decContext: CsvRowDecoder[ImagingContext, String], 
            decCentroid: CsvRowDecoder[Centroid[Double], String],
            decBox: CsvRowDecoder[BoundingBox, String],
            decNuc: CellDecoder[NucleusNumber]
        ): CsvRowDecoder[InputRecord, String] = new:
            override def apply(row: RowF[Some, String]): DecoderResult[InputRecord] = 
                val spotNel = summon[CsvRowDecoder[IndexedDetectedSpot, String]](row)
                    .leftMap(e => e.getMessage)
                    .toValidatedNel
                val mergeInputsNel: ValidatedNel[String, Set[RoiIndex]] = 
                    MergeContributorsColumnNameForAssessedRecord.from(row)
                val nucNel: ValidatedNel[String, Option[NucleusNumber]] = ???
                (spotNel, mergeInputsNel, nucNel)
                    .mapN{ (spot, maybeMergeIndices, maybeNucNum) => 
                        InputRecord(spot.index, spot.context, spot.centroid, spot.box, maybeMergeIndices, maybeNucNum)
                    }
                    .toEither
                    .leftMap{ messages => 
                        DecoderError(s"${messages.length} error(s) reading row ($row):\n${messages.mkString_("\n")}")
                    }
    end InputRecord
end AssignTraceIds
