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
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.TraceIdDefinitionAndFiltrationRule
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
        import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2.syntax.toList
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

    private def computeNeighborsGraph(rules: NonEmptyList[TraceIdDefinitionAndFiltrationRule])(records: List[InputRecord]): SimplestGraph[RoiIndex] = 
        import ProximityComparable.proximal
        val lookupProximity: Map[(ImagingTimepoint, ImagingTimepoint), ProximityComparable[InputRecord]] = 
            definePairwiseDistanceThresholds(rules)
                .view
                .mapValues{ dt => DistanceThreshold.defineProximityPointwise(dt)((_: InputRecord).centroid.asPoint) }
                .toMap
        val edgeEndpoints: Set[(RoiIndex, RoiIndex)] = 
            records.groupBy(r => r.context.fieldOfView -> r.context.channel) // Only merge ROIs from the same context (FOV, channel).
                .values
                .flatMap(_.combinations(2).flatMap{
                    case r1 :: r2 :: Nil => 
                        given ProximityComparable[InputRecord] = lookupProximity(r1.context.timepoint -> r2.context.timepoint)
                        (r1 `proximal` r2).option{ r1.index -> r2.index }
                    case notPair => throw new Exception(s"Got ${notPair.length} element(s) when taking pairs!")
                })
                .toSet
        buildSimpleGraph(records.map(_.index).toSet, edgeEndpoints)
    
    private def sieveRecords(rules: NonEmptyList[TraceIdDefinitionAndFiltrationRule])(records: List[InputRecord]): (List[InputRecord], List[(InputRecord, TraceId)]) = 
        val graph = computeNeighborsGraph(rules)(records)
        // TODO: implementation.
        ???

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
    )

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
