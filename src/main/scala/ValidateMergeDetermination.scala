package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.NotGiven
import cats.Alternative
import cats.data.{ NonEmptyList, NonEmptySet, ValidatedNel }
import cats.effect.IO
import cats.effect.unsafe.implicits.global // for IORuntime
import cats.syntax.all.*
import mouse.boolean.*
import fs2.data.csv.*
import fs2.data.text.utf8.*
import io.github.iltotore.iron.cats.given
import io.github.iltotore.iron.constraint.collection.given
import com.typesafe.scalalogging.StrictLogging
import scopt.*

import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2
import at.ac.oeaw.imba.gerlich.gerlib.geometry.{ Centroid, DistanceThreshold, EuclideanDistance }
import at.ac.oeaw.imba.gerlich.gerlib.graph.buildSimpleGraph
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{ 
    ColumnName, 
    ColumnNameLike, 
    getCsvRowDecoderForSingleton, 
    getCsvRowDecoderForTuple2, 
    readCsvToCaseClasses,
}
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.FieldOfViewColumnName
import at.ac.oeaw.imba.gerlich.gerlib.numeric.{ NonnegativeInt, NonnegativeReal }
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames.{
    RoiIndexColumnName,
    TooCloseRoisColumnName,
    TraceIdColumnName, 
    TracePartnersColumName,
}
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.gerlib.imaging.PositionName

/**
 * Check that the determination of the connected components of merge partners are as expected
 */
object ValidateMergeDetermination extends ScoptCliReaders, StrictLogging:
    val ProgramName = "ValidateMergeDetermination"

    /* Type aliases */
    type Head = String
    type Row = RowF[Some, Head]
    type RowDec[A] = CsvRowDecoder[A, Head]
    type IdPair = (RoiIndex, RoiIndex)
    type MergePartners = NonEmptySet[RoiIndex]
    type ParseTarget[R <: InputRecord] = (R, Option[MergePartners])
    type ValidE[Error] = Either[Error, Unit]
    type SplitGroup = AtLeast2[Set, NonEmptySet[RoiIndex]]
    type NoBadUngrouped = Either[Unit, Unit]
    type GroupedResult = ValidE[NonEmptyList[SplitGroup]]
    type UngroupedResult = Either[NonEmptyList[(IdPair, EuclideanDistance)], NoBadUngrouped]

    case class CliConfig(
        inputFile: os.Path = null, // required
        mergeType: MergeType = null, // required
        differentTimepointDistanceThreshold: Option[EuclideanDistance.Threshold] = None, 
        sameTimepointDistanceThreshold: Option[EuclideanDistance.Threshold] = None
    )

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = 
        import parserBuilder.*
        
        given Read[EuclideanDistance.Threshold] = 
            summon[Read[NonnegativeReal]].map(EuclideanDistance.Threshold.apply)

        val parser = OParser.sequence(
            programName(ProgramName),
            head(ProgramName, BuildInfo.version), 
            opt[os.Path]('I', "inputFile")
                .required()
                .action((f, c) => c.copy(inputFile = f))
                .validate(f => os.isFile(f).either(s"Alleged input file isn't an extant file: $f", ()))
                .text("Path to the file containing ROI merge determination information to validate"),
            opt[MergeType]('M', "mergeType")
                .required()
                .action((m, c) => c.copy(mergeType = m))
                .text(s"Type of merge determination file being analyzed; choose from: ${MergeType.values.mkString(", ")}"),
            opt[EuclideanDistance.Threshold]("differentTimepointDistanceThreshold")
                .action((t, c) => c.copy(differentTimepointDistanceThreshold = t.some))
                .text("Distance threshold which was used to define the merge partners for ROIs from DIFFERENT timepoints"), 
            opt[EuclideanDistance.Threshold]("sameTimepointDistanceThreshold")
                .action((t, c) => c.copy(sameTimepointDistanceThreshold = t.some))
                .text("Distance threshold which was used to define the merge partners for ROIs from the SAME timepoint"), 
            checkConfig{ c => 
                (c.mergeType, c.sameTimepointDistanceThreshold, c.differentTimepointDistanceThreshold) match {
                    case (MergeType.SameTimepoint, same, diff) => 
                        List(
                            same.isEmpty.option("the same-timepoint distance threshold is required"), 
                            diff.nonEmpty.option("the different-timepoint distance threshold is prohibited")
                        )
                        .flatten
                        .map("For same-timepoint ROI merge determination analysis, " ++ _)
                        .toNel
                        .fold(success){ messages => failure(messages.mkString_("; ")) }
                    case (MergeType.DifferentTimepoint, _, diff) => 
                        if diff.nonEmpty then success 
                        else failure("For different-timepoint ROI merge determination analysis, different-timepoint threshold is required.")
                }
            }
        )

        def getDecoderAndThreshold(
            mergeType: MergeType, 
            diffTimeThreshold: Option[EuclideanDistance.Threshold], 
            sameTimeThreshold: Option[EuclideanDistance.Threshold],
        ): (RowDec[InputRecord], EuclideanDistance.Threshold) = 
            import SameTimepointRecord.given
            import DifferentTimepointRecord.given

            mergeType match {
                case MergeType.SameTimepoint => 
                    val dec: RowDec[InputRecord] = new:
                        override def apply(row: Row): DecoderResult[InputRecord] = summon[RowDec[SameTimepointRecord]](row)
                    (dec, sameTimeThreshold.getOrElse{ 
                        throw new RuntimeException("Merge type is same-timepoint, but there's no same-timepoint threshold!") 
                    })
                case MergeType.DifferentTimepoint => 
                    val dec: RowDec[InputRecord] = new:
                        override def apply(row: Row): DecoderResult[InputRecord] = summon[RowDec[DifferentTimepointRecord]](row)
                    (dec, diffTimeThreshold.getOrElse{
                        throw new RuntimeException("Merge type is different-timepoint, but there's no different-timepoint threshold!") 
                    })
            }

        OParser.parse(parser, args, CliConfig()) match {
            case None => 
                // CLI parser gives error message.
                throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help")
            case Some(opts) => 
                given RowDec[Option[MergePartners]] = 
                    given CellDecoder[Option[MergePartners]] with
                        override def apply(cell: String): DecoderResult[Option[MergePartners]] = cell match {
                            case "" => None.asRight
                            case _ => cell.split(" ")
                                .toList
                                .traverse{ s => NonnegativeInt.parse(s).map(RoiIndex.apply) }
                                .flatMap(_.toNel.toRight("Empty merge partners list!").map(_.toNes))
                                .bimap(msg => new DecoderError(msg), _.some)
                        }    
                    getCsvRowDecoderForSingleton(ColumnName[Option[MergePartners]](opts.mergeType.column))
                
                val (decRow, thresholdForGrouped) = getDecoderAndThreshold(
                    mergeType = opts.mergeType,
                    diffTimeThreshold = opts.differentTimepointDistanceThreshold, 
                    sameTimeThreshold = opts.sameTimepointDistanceThreshold,
                )
                
                // Create the decoder for the pair of record and (optional) nonempty merge partners collection.
                given RowDec[(InputRecord, Option[MergePartners])] = getCsvRowDecoderForTuple2(using decRow, summon[RowDec[Option[MergePartners]]])
                
                val parseAndDigestGrouped: IO[(List[InputRecord], List[SplitGroup])] = opts.mergeType match {
                    case MergeType.SameTimepoint => 
                        type Record = SameTimepointRecord
                        readInputFile[Record](opts.inputFile)
                            .map(splitRecords[Record])
                            .map{ (ungrouped, grouped) => ungrouped -> findSplitGroups(thresholdForGrouped)(grouped) }
                    case MergeType.DifferentTimepoint => 
                        type Record = DifferentTimepointRecord
                        readInputFile[Record](opts.inputFile)
                            .map(splitRecords[Record])
                            .map{ (ungrouped, grouped) => ungrouped -> findSplitGroups(thresholdForGrouped)(grouped) }
                }

                val parsedAndDigested: IO[(UngroupedResult, GroupedResult)] = 
                    parseAndDigestGrouped.map{ (ungrouped, splitGroups) => 
                        val ugRes: UngroupedResult = opts.sameTimepointDistanceThreshold match {
                            case Some(thresholdForUngrouped) => 
                                checkUngrouped(thresholdForUngrouped)(ungrouped)
                            case None => NoBadUngrouped.trivial.asRight
                        }
                        val grRes: GroupedResult = splitGroups.toNel.toLeft(())
                        ugRes -> grRes
                    }

                parsedAndDigested.flatMap{ (ungroupedResult, groupedResult) => 
                    for {
                        _ <- IO{
                            val msg = ungroupedResult match {
                                case Left(pairs) => s"${pairs.length} pair(s) of proximal records: $pairs"
                                case Right(value) => value.fold(
                                    Function.const{ "No analysis of ungrouped records was done." }, 
                                    Function.const{ "All ungrouped records were correctly singleton" },
                                )
                            }
                            logger.info(msg)
                        }
                        _ <- IO{
                            val msg = groupedResult match {
                                case Left(splitGroups) => s"${splitGroups.length} split group(s): $splitGroups"
                                case Right(value) => "All good with grouped records :)"
                            }
                            logger.info(msg)
                        }
                    } yield ()
                }
                .unsafeRunSync()
        }

    /** The possibilities for the type of merge determination to be checking */
    enum MergeType(val column: String):
        case SameTimepoint extends MergeType(TooCloseRoisColumnName.value)
        case DifferentTimepoint extends MergeType(TracePartnersColumName.value)

    object MergeType:
        given Read[MergeType] = Read.reads{ s => 
            s.toLowerCase() match {
                case "same" => MergeType.SameTimepoint
                case "different" => MergeType.DifferentTimepoint
                case _ => throw new IllegalArgumentException(s"$s does not correspond to a merge type.")
            }
        }

    sealed trait InputRecord:
        def fieldOfView: PositionName
        def roiId: RoiIndex
        def timepoint: ImagingTimepoint
        def centroid: Centroid[Double]
    
    object InputRecord:
        extension [R <: InputRecord](r: R)
            infix def dist(that: R): EuclideanDistance = 
                import Centroid.asPoint
                EuclideanDistance.between((_: R).centroid.asPoint)(r, that)
        
        given RowDec[InputRecord] with
            override def apply(row: Row): DecoderResult[InputRecord] = 
                val fovNel = ColumnName[PositionName](FieldOfViewColumnName.value).from(row)
                val roiIdNel = RoiIndexColumnName.from(row)
                val timeNel = ColumnName[ImagingTimepoint]("timepoint").from(row)
                val centroidNel = parseCentroid(row)
                (fovNel, roiIdNel, timeNel, centroidNel)
                    .mapN{ (p, i, t, c) => 
                        new InputRecord:
                            override def fieldOfView = p
                            override def roiId = i
                            override def timepoint = t
                            override def centroid = c
                    }
                    .toDecoderResult("decoding base record for merge determination validation")
    end InputRecord

    final case class DifferentTimepointRecord(
        fieldOfView: PositionName,
        roiId: RoiIndex, 
        timepoint: ImagingTimepoint,
        centroid: Centroid[Double],
        traceId: TraceId, 
    ) extends InputRecord

    object DifferentTimepointRecord:
        def fromBaseRecord = (base: InputRecord, tid: TraceId) => 
            DifferentTimepointRecord(base.fieldOfView, base.roiId, base.timepoint, base.centroid, tid)
        
        given RowDec[DifferentTimepointRecord] = new:
            override def apply(row: Row): DecoderResult[DifferentTimepointRecord] = 
                val recNel = summon[RowDec[InputRecord]](row)
                    .leftMap(_.getMessage)
                    .toValidatedNel
                val xNel = TraceIdColumnName.from(row)
                (recNel, xNel).mapN(fromBaseRecord).toDecoderResult("decoding different-timepoint merge record")
    end DifferentTimepointRecord

    final case class SameTimepointRecord(
        fieldOfView: PositionName,
        roiId: RoiIndex, 
        timepoint: ImagingTimepoint, 
        centroid: Centroid[Double],
    ) extends InputRecord

    object SameTimepointRecord:
        def fromBaseRecord = (base: InputRecord) => 
            SameTimepointRecord(base.fieldOfView, base.roiId, base.timepoint, base.centroid)
        
        given decoderForRecord(using decBase: RowDec[InputRecord]): RowDec[SameTimepointRecord] = decBase.map(fromBaseRecord)
    end SameTimepointRecord

    object NoBadUngrouped:
        def trivial: NoBadUngrouped = ().asLeft[Unit]
        def nontrivial: NoBadUngrouped = ().asRight[Unit]

    def findSplitGroups[R <: InputRecord : [R] =>> NotGiven[R =:= InputRecord]](threshold: EuclideanDistance.Threshold)(groups: List[AtLeast2[Set, R]]): List[SplitGroup] = 
        groups.flatMap{ g => validateGroupIsSingleConnectedComponent(threshold)(g).swap.toOption }

    def readInputFile[R <: InputRecord](f: os.Path)(using NotGiven[R =:= InputRecord], RowDec[R], RowDec[Option[MergePartners]]): IO[List[ParseTarget[R]]] = 
        logger.info(s"Processing file: $f")
        given RowDec[ParseTarget[R]] = getCsvRowDecoderForTuple2
        readCsvToCaseClasses[ParseTarget[R]](f)

    def splitRecords[R <: InputRecord : [R] =>> NotGiven[R =:= InputRecord]](records: List[(R, Option[MergePartners])]): (List[R], List[AtLeast2[Set, R]]) = 
        val lookup: Map[RoiIndex, R] = records.map(_._1).map(r => r.roiId -> r).toMap
        Alternative[List].separate(
            records.map { (r, partnersOpt) => 
                partnersOpt.toRight(r).map{ partners => AtLeast2.unsafe((r :: partners.toList.map(lookup)).toSet) }
            }
        )

    def checkUngrouped(threshold: EuclideanDistance.Threshold): List[InputRecord] => Either[NonEmptyList[(IdPair, EuclideanDistance)], NoBadUngrouped] = 
        _.groupBy(r => r.fieldOfView -> r.timepoint)
            .values
            .flatMap(_
                .combinations(2)
                .flatMap(pairFuncOntoList(ifProximal(threshold){ (r1, r2, d) => (r1.roiId -> r2.roiId) -> d }))
            )
            .toList
            .toNel
            .toLeft(NoBadUngrouped.nontrivial)

    // TODO: since the "k" (number of items in each selection) is often known statically, e.g. .combinations(2), 
    //       gerlib could contain a utility which produces a collection C[*] where C is determined by the type 
    //       of collection on which the .combinations call is invoked...
    //       extension [C[*] <: Iterable[*]]
    //           def pairs[A](as: C[A]): (A, A) = ???
    //           def sub2[A](as: C[A]): C[A] :| Eq[Length[2]] = ???
    def pairFuncOntoList[A, B](f: (A, A) => B): List[A] => B = _ match {
        case a1 :: a2 :: Nil => f(a1, a2)
        case as => throw new RuntimeException(s"Got ${as.length} elements when taking pairs of 2")
    }

    def ifProximal[R <: InputRecord, O](threshold: EuclideanDistance.Threshold)(f: (R, R, EuclideanDistance) => O): (R, R) => Option[O] = 
        val dist: (R, R) => EuclideanDistance = EuclideanDistance.between(_.centroid.asPoint)
        (r1, r2) => dist(r1, r2).some.flatMap{ d => d.lessThan(threshold).option(f(r1, r2, d)) }

    def emitIdPairIfProximal[R <: InputRecord](threshold: EuclideanDistance.Threshold): List[R] => Option[IdPair] = 
        pairFuncOntoList(ifProximal(threshold){ (r1: R, r2: R, d: EuclideanDistance) => r1.roiId -> r2.roiId })

    def parseCentroid(row: Row): ValidatedNel[String, Centroid[Double]] = 
        summon[RowDec[Centroid[Double]]](row)
            .leftMap{ e => NonEmptyList.one(e.getMessage) }
            .toValidated

    def validateGroupIsSingleConnectedComponent[R <: InputRecord](threshold: EuclideanDistance.Threshold): AtLeast2[Set, R] => ValidE[SplitGroup] = 
        import AtLeast2.syntax.*
        group => 
            val nodes = group.toList.map(_.roiId).toSet
            val edges: Set[IdPair] = group.toList
                .combinations(2)
                .flatMap(emitIdPairIfProximal(threshold))
                .toSet
            val graph = buildSimpleGraph(nodes, edges)
            AtLeast2.either(
                graph.strongComponentTraverser()
                    .map(_.nodes
                        .map(_.outer)
                        .toList
                        .toNel
                        .getOrElse(throw new RuntimeException("A connected component contains no nodes!"))
                        .toNes
                    )
                    .toSet
                )
                .swap
                .map(_ => ())

    extension [A](afterApply: ValidatedNel[String, A])
        def toDecoderResult(context: String): DecoderResult[A] = 
            afterApply.toEither.leftMap{ messages => 
                DecoderError(s"${messages.length} error(s) ($context): ${messages.mkString_("; ")}") 
            }

end ValidateMergeDetermination
