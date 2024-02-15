package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import upickle.default.*

import cats.*
import cats.data.*
import cats.data.Validated.{ Invalid, Valid }
import cats.syntax.all.*
import mouse.boolean.*
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.safeExtract

/** A round of imaging during an experiment */
sealed trait ImagingRound:
    def name: String
    def time: Timepoint
end ImagingRound

sealed trait FishImagingRound extends ImagingRound:
    def probe: ProbeName
    def repeat: Option[PositiveInt]
end FishImagingRound

final case class BlankImagingRound(name: String, time: Timepoint) extends ImagingRound

final case class LocusImagingRound(name: String, time: Timepoint, probe: ProbeName, repeat: Option[PositiveInt]) extends FishImagingRound

object LocusImagingRound:
    def apply(time: Timepoint, probe: ProbeName): LocusImagingRound = apply(None, time, probe, None)
    def apply(time: Timepoint, probe: ProbeName, repeat: PositiveInt): LocusImagingRound = apply(None, time, probe, repeat.some)
    def apply(maybeName: Option[String], time: Timepoint, probe: ProbeName, maybeRepeat: Option[PositiveInt]): LocusImagingRound = 
        val name = maybeName.getOrElse(probe.get ++ maybeRepeat.fold("")(rep => s"_repeat${rep.show}"))
        new LocusImagingRound(name, time, probe, maybeRepeat)
end LocusImagingRound

final case class RegionalImagingRound(name: String, time: Timepoint, probe: ProbeName) extends FishImagingRound:
    override final def repeat: Option[PositiveInt] = None

end RegionalImagingRound

object RegionalImagingRound:
    def apply(time: Timepoint, probe: ProbeName): RegionalImagingRound = apply(None, time, probe)
    def apply(maybeName: Option[String], time: Timepoint, probe: ProbeName): RegionalImagingRound = 
        new RegionalImagingRound(maybeName.getOrElse(probe.get), time, probe)
end RegionalImagingRound

final case class ImagingSequence private(rounds: NonEmptyList[ImagingRound])

/** Smart constructors and tools for working with sequences of imaging rounds */
object ImagingSequence:
    private type InvalidOr[A] = ValidatedNel[String, A]
    private def extractNameAndTime(json: ujson.Value): (InvalidOr[String], InvalidOr[Timepoint]) = 
        val maybeName = safeExtract("name", identity)(json)
        val maybeTime = (Try{ json("time").int }.toEither.leftMap(_.getMessage) >>= Timepoint.fromInt).toValidatedNel
        maybeName -> maybeTime

    private def validateNoExtraKeys(validKeys: Set[String], context: String)(json: ujson.Value): InvalidOr[Unit] = 
        (json.obj.keySet.toSet -- validKeys)
            .toList
            .toNel
            .toLeft(())
            .leftMap(extra => s"Extra key(s) for decoding $context: ${extra.mkString_(", ")}")
            .toValidatedNel

    given rwForBlankImagingRound: ReadWriter[BlankImagingRound] = readwriter[ujson.Value].bimap(
        round => ujson.Obj("name" -> ujson.Str(round.name), "time" -> ujson.Num(round.time.get)),
        json => 
            val (maybeName, maybeTime) = extractNameAndTime(json)
            val noExtraKeys = validateNoExtraKeys(Set("name", "time"), "blank imaging round")(json)
            (maybeName, maybeTime, noExtraKeys).tupled match {
                case Invalid(errors) => 
                    throw new ujson.Value.InvalidData(json, s"Error(s) decoding as : ${errors.mkString_(", ")}")
                case Valid((name, time, _)) => BlankImagingRound(name, time)
            }
    )

    given rwForFishImagingRound: ReadWriter[FishImagingRound] = readwriter[ujson.Value].bimap(
        {
            case round: RegionalImagingRound => ujson.Obj(
                "name" -> round.name, 
                "time" -> ujson.Num(round.time.get), 
                "isRegional" -> ujson.Bool(true)
                )
            case round: LocusImagingRound => ujson.Obj(
                "name" -> round.name, 
                "time" -> ujson.Num(round.time.get), 
                "repeat" -> round.repeat.fold(ujson.Null)(n => ujson.Num(n))
            )
        }, 
        json => (for {
            probe <- Try{ json("probe").str }.toEither.bimap(e => NonEmptyList.one(e.getMessage), ProbeName.apply)
            (time, maybeName) <- {
                val (maybeName, maybeTime) = extractNameAndTime(json)
                val noExtraKeys = validateNoExtraKeys(Set("name", "time", "isRegional", "repeat"), "FISH imaging round")(json)
                (maybeTime, noExtraKeys).mapN((time, _) => time -> maybeName.toOption).toEither
            }
            obj = json.obj
            (isRegional, maybeRepNum) <- {
                val reg = (obj.get("isRegional") match {
                    case None => false.asRight
                    case Some(isRegJson) => Try(isRegJson.bool).toEither.leftMap(e => s"Non-Boolean value for regionality key! ${e.getMessage}")
                }).toValidatedNel
                val rep = (obj.get("repeat") match {
                    case None => None.asRight
                    case Some(rawRepJson) => Try(rawRepJson.int)
                        .toEither
                        .leftMap(e => s"Non-integral value for repeat index! ${e.getMessage}")
                        .flatMap{ z => PositiveInt.either(z).bimap(errMsg => s"Illegal repeat index ($z): $errMsg", _.some) }
                }).toValidatedNel
                (reg, rep).tupled.toEither
            }
            result <- 
                if isRegional 
                then maybeRepNum match {
                    case None => (RegionalImagingRound(maybeName, time, probe)).asRight
                    case Some(_) => NonEmptyList.one(s"Repeat declared for REGIONAL imaging round (time ${time.show})!").asLeft
                }
                else LocusImagingRound(maybeName, time, probe, maybeRepNum).asRight
        } yield result).fold(errors => throw new ujson.Value.InvalidData(json, s"Error(s) decoding as FISH imaging round: ${errors.mkString_(", ")}"), identity)
    )
    
    def fromRounds(maybeRounds: List[ImagingRound]): ErrMsgsOr[ImagingSequence] = maybeRounds.toNel
        .toRight("Can't build an imaging sequence from empty collection of rounds!")
        .flatMap{ rounds => 
            val ordByTime = rounds.sortBy(_.time)
            val times = ordByTime.map(_.time.get.toInt).toList
            for {
                _ <- (times === (0 until ordByTime.length).toList).either(
                    s"Ordered timepoints for imaging rounds don't form contiguous sequence from 0 up to length (${times.length}): ${times.mkString(", ")}", 
                    ()
                    )
                _ <- rounds.groupBy(_.name).view.mapValues(_.size).filter(_._2 > 1).toList match {
                    case Nil => ().asRight
                    case namesHisto => s"Repeated names in imaging round sequence! ${namesHisto}".asLeft
                }
            } yield ordByTime
        }
        .bimap(NonEmptyList.one, ImagingSequence.apply)
