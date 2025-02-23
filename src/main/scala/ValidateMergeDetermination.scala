package at.ac.oeaw.imba.gerlich.looptrace

import cats.Alternative
import cats.data.{ NonEmptyList, NonEmptySet }
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
import at.ac.oeaw.imba.gerlich.gerlib.geometry.{ Centroid, EuclideanDistance }
import at.ac.oeaw.imba.gerlich.gerlib.graph.buildSimpleGraph
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{ ColumnName, getCsvRowDecoderForSingleton, getCsvRowDecoderForTuple2, readCsvToCaseClasses }
import at.ac.oeaw.imba.gerlich.gerlib.numeric.{ NonnegativeInt, NonnegativeReal }
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.csv.ColumnNames
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo

/**
 * Check that the determination of the connected components of merge partners are as expected
 */
object ValidateMergeDetermination extends ScoptCliReaders, StrictLogging:
    val ProgramName = "ValidateMergeDetermination"

    type Head = String
    type Row = RowF[Some, Head]
    type RowDec[A] = CsvRowDecoder[A, Head]
    type ParseTarget = (InputRecord, Option[MergePartners])
    type FromInputPair[O] = Function2[InputRecord, InputRecord, O]
    type IdPair = (RoiIndex, RoiIndex)
    type MergePartners = NonEmptySet[RoiIndex]
    type ValidE[Error] = Either[Error, Unit]

    case class CliConfig(
        inputFile: os.Path = null, // required
        threshold: NonnegativeReal = NonnegativeReal(0), // required
    )

    case class InputRecord(roiId: RoiIndex, traceId: TraceId, centroid: Centroid[Double])

    object InputRecord:
        extension (r: InputRecord)
            infix def dist(that: InputRecord): EuclideanDistance = 
                import Centroid.asPoint
                EuclideanDistance.between[InputRecord](_.centroid.asPoint)(r, that)
    end InputRecord

    val parserBuilder = OParser.builder[CliConfig]

    def validateGroupIsSingleConnectedComponent(threshold: NonnegativeReal): AtLeast2[Set, InputRecord] => ValidE[AtLeast2[Set, Set[RoiIndex]]] = 
        import AtLeast2.syntax.*
        group => 
            val nodes = group.toList.map(_.roiId).toSet
            val edges: Set[IdPair] = group.toList
                .combinations(2)
                .flatMap(fun2List(ifProximal(threshold){ (r1: InputRecord, r2: InputRecord, d: EuclideanDistance) => r1.roiId -> r2.roiId }))
                .toSet
            val graph = buildSimpleGraph(nodes, edges)
            AtLeast2.either(graph.strongComponentTraverser().map(_.nodes.map(_.outer)).toSet)
                .swap
                .map(_ => ())

    def ifProximal[B](threshold: NonnegativeReal)(f: (InputRecord, InputRecord, EuclideanDistance) => B): FromInputPair[Option[B]] = (r1, r2) => 
        (r1 `dist` r2).some.flatMap(d => (d.get < threshold).option(f(r1, r2, d)))

    def fun2List[A, B](f: (A, A) => B): List[A] => B = _ match {
        case a1 :: a2 :: Nil => f(a1, a2)
        case as => throw new RuntimeException(s"Got ${as.length} elements when taking pairs of 2")
    }

    def main(args: Array[String]): Unit = 
        import parserBuilder.*
        
        val parser = OParser.sequence(
            programName(ProgramName),
            head(ProgramName, BuildInfo.version), 
            opt[os.Path]('I', "inputFile")
                .required()
                .action((f, c) => c.copy(inputFile = f))
                .validate(f => os.isFile(f).either(s"Alleged input file isn't an extant file: $f", ()))
                .text("Path to the file containing ROI merge determination information to validate"),
            opt[NonnegativeReal]('D', "distanceThreshold")
                .required()
                .action((d, c) => c.copy(threshold = d))
                .text("Distance threshold which was used to define the merge partners")
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => 
                // CLI parser gives error message.
                throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help")
            case Some(opts) => 
                given RowDec[ParseTarget] = 
                    given RowDec[InputRecord] with
                        override def apply(row: Row): DecoderResult[InputRecord] = 
                            val roiIdNel = ColumnNames.RoiIndexColumnName.from(row)
                            val traceIdNel = ColumnNames.TraceIdColumnName.from(row)
                            val centroidNel = summon[RowDec[Centroid[Double]]](row)
                                .leftMap{ e => NonEmptyList.one(e.getMessage) }
                                .toValidated
                            (roiIdNel, traceIdNel, centroidNel).mapN(InputRecord.apply)
                                .toEither
                                .leftMap{ messages => 
                                    DecoderError(s"${messages.length} error(s) decoding input record: ${messages.mkString_("; ")}") 
                                }
                    given CellDecoder[Option[MergePartners]] with
                        override def apply(cell: String): DecoderResult[Option[MergePartners]] = cell match {
                            case "" => None.asRight
                            case _ => cell.split(" ")
                                .toList
                                .traverse{ s => NonnegativeInt.parse(s).map(RoiIndex.apply) }
                                .flatMap(_.toNel.toRight("Empty merge partners list!").map(_.toNes))
                                .bimap(msg => new DecoderError(msg), _.some)
                        }
                    given RowDec[Option[MergePartners]] = 
                        val col = ColumnName[Option[MergePartners]](ColumnNames.TracePartnersColumName.value)
                        getCsvRowDecoderForSingleton(col)
                    getCsvRowDecoderForTuple2

                logger.info(s"Processing file: ${opts.inputFile}")
                readCsvToCaseClasses[ParseTarget](opts.inputFile)
                    .map{ records => 
                        val lookup: Map[RoiIndex, InputRecord] = records.map(_._1).map(r => r.roiId -> r).toMap
                        records.flatMap{ (r, maybePartners) => 
                            maybePartners.map{ partners => 
                                AtLeast2.unsafe((r :: partners.toList.map(lookup)).toSet)
                            }
                        }
                    }
                    .map(_.flatMap{ g => validateGroupIsSingleConnectedComponent(opts.threshold)(g).swap.toOption }
                        .toNel
                        .toLeft(())
                    )
                    .flatMap{
                        case Left(badGroups) => IO{
                            logger.info("Bad groups!")
                            badGroups.toList.foreach(g => logger.info(g.toString))
                        }
                        case Right(()) => IO{
                            logger.info("No problems for grouped records :)")
                        }
                    }
                    .unsafeRunSync()
        }

end ValidateMergeDetermination
