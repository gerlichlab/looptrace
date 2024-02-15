package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import upickle.default.*

import cats.data.*
import cats.data.Validated.{ Invalid, Valid }
import cats.syntax.all.*
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.looptrace.syntax.*

/** A sequence of FISH and blank imaging rounds, constituting a microscopy experiment */
final case class ImagingSequence private(rounds: NonEmptyList[ImagingRound]):
    lazy val _lookup = rounds.map(r => r.name -> r).toNem
    def get(name: String): Option[ImagingRound] = _lookup(name)
end ImagingSequence

/** Smart constructors and tools for working with sequences of imaging rounds */
object ImagingSequence:
    private type InvalidOr[A] = ValidatedNel[String, A]

    /** Try to parse a single imaging round instance from a raw JSON data mapping */
    def parse1(data: Map[String, ujson.Value]): ErrMsgsOr[ImagingRound] = {
        val keys = data.keySet
        val timeNel = (data.get("time").toRight("Missing timepoint!") >>= parseTimeValue).toValidatedNel
        val isRegionalNel = extractDefaultFalse("isRegional")(data).toValidatedNel
        val isBlankNel = extractDefaultFalse("isBlank")(data).toValidatedNel
        val nameOptNel = extractOptionalString("name")(data).toValidatedNel
        val probeOptNel = extractOptionalString("probe")(data).toValidatedNel
        val repOptNel = data.get("repeat").traverse(parseRepeatValue).toValidatedNel
        val noExtraKeysNel = validateNoExtraKeys(Set("time", "probe", "name", "isRegional", "isBlank"), "image round")(data)
        (timeNel, isRegionalNel, isBlankNel, nameOptNel, probeOptNel, repOptNel, noExtraKeysNel).tupled.toEither.flatMap{
            case (time, isRegional, isBlank, nameOpt, probeOpt, repOpt, _) => 
                val regRepNel = (isRegional && repOpt.nonEmpty).toValidatedNel(s"Regional round cannot be a repeat!")
                val isBlankIsRegionalNel = (!isRegional || !isBlank).toValidatedNel("Image round can't be both blank and regional!")
                val isBlankProbeNel = ((probeOpt, isBlank) match {
                    /* Ensure that blank spec and probe spec are compatible. */
                    case (Some(_), true) => "Blank frame cannot have probe specified!".asLeft
                    case (None, false) => "Probe is required when a round isn't blank!".asLeft
                    case _ => ().asRight
                }).toValidatedNel
                val nameProbeRepNel = ((nameOpt, probeOpt) match {
                /* Ensure that combination of name, probe, and repeat are compatible. */
                    case (None, None) => "At least probe or name is always required for image round.".asLeft
                    case (None, Some(probe)) => (probe ++ repOpt.fold("")(rep => s"_repeat${rep.show}"), ProbeName(probe)).asRight
                    case (Some(name), _) => (name, ProbeName(probeOpt.getOrElse(name))).asRight
                }).toValidatedNel
                (regRepNel, isBlankIsRegionalNel, isBlankProbeNel, nameProbeRepNel).mapN{ 
                    case (_, _, _, (name, probe)) => 
                        if isBlank then BlankImagingRound(name, time)
                        else if isRegional then RegionalImagingRound(name, time, probe)
                        else LocusImagingRound(name, time, probe, repOpt)
                }.toEither
        }
    }

    private def preprocess1(data: ujson.Value): Either[String, Map[String, ujson.Value]] = 
        Try(data.obj).toEither.bimap(_.getMessage, _.toMap)

    private def parseTimeValue(v: ujson.Value): Either[String, Timepoint] = 
        Try(v.int).toEither.leftMap(e => s"Illegal value for time! ${e.getMessage}") 
            >>= NonnegativeInt.either 
            >> Timepoint.apply

    private def parseRepeatValue(v: ujson.Value): Either[String, PositiveInt] = 
        Try(v.int).toEither.leftMap(e => s"Illegal value for repeat! ${e.getMessage}") 
            >>= PositiveInt.either

    private def extractDefaultFalse(key: String)(data: Map[String, ujson.Value]): Either[String, Boolean] = 
        data.get(key) match {
            case None => false.asRight
            case Some(v) => Try(v.bool).toEither.leftMap(e => s"Illegal value for '$key'! ${e.getMessage}")
        }

    private def extractOptionalString(key: String)(data: Map[String, ujson.Value]): Either[String, Option[String]] = 
        data.get(key).traverse{ v => Try(v.str).toEither.leftMap(e => s"Illagel value for '$key'! ${e.getMessage}") }

    extension (p: Boolean)
        private def toValidatedNel(msg: String): ValidatedNel[String, Unit] = p.either(msg, ()).toValidatedNel
 
    private def validateNoExtraKeys(validKeys: Set[String], whatToDecode: String)(data: Map[String, ujson.Value]): InvalidOr[Unit] = {
        (data.keySet.toSet -- validKeys)
            .toList
            .toNel
            .toLeft(())
            .leftMap(extra => s"Extra key(s) for decoding $whatToDecode: ${extra.mkString_(", ")}")
            .toValidatedNel
    }

    private def blankRoundToJson(round: BlankImagingRound) = 
        ujson.Obj("name" -> ujson.Str(round.name), "time" -> ujson.Num(round.time.get))

    private def locusRoundToJson(round: LocusImagingRound) = ujson.Obj(
        "name" -> round.name, 
        "time" -> ujson.Num(round.time.get), 
        "repeat" -> round.repeat.fold(ujson.Null)(n => ujson.Num(n))
        )

    private def regionalRoundToJson(round: RegionalImagingRound) = ujson.Obj(
        "name" -> round.name, 
        "time" -> ujson.Num(round.time.get), 
        "isRegional" -> ujson.Bool(true)
        )

    private def roundToJsonObject(fishRound: FishImagingRound): ujson.Obj = fishRound match {
        case round: LocusImagingRound => locusRoundToJson(round)
        case round: RegionalImagingRound => regionalRoundToJson(round)
    }

    private def roundToJsonObject(imagingRound: ImagingRound): ujson.Obj = imagingRound match {
        case round: BlankImagingRound => blankRoundToJson(round)
        case round: FishImagingRound => roundToJsonObject(round)
    }

    def rwForImagingRound: ReadWriter[ImagingRound] = readwriter[ujson.Value].bimap(
        roundToJsonObject, 
        json => parse1(json.obj.toMap)
            .fold(es => throw ujson.Value.InvalidData(json, s"Error(s) decoding imaging round: ${es.mkString_(", ")}"), identity)
    )
    
    def fromRounds(maybeRounds: List[ImagingRound]): ErrMsgsOr[ImagingSequence] = maybeRounds.toNel
        .toRight(NonEmptyList.one("Can't build an imaging sequence from empty collection of rounds!"))
        .flatMap{ rounds => 
            val ordByTime = rounds.sortBy(_.time)
            val times = ordByTime.map(_.time.get.toInt).toList
            val timesNel = (times === (0 until ordByTime.length).toList).toValidatedNel(
                s"Ordered timepoints for imaging rounds don't form contiguous sequence from 0 up to length (${times.length}): ${times.mkString(", ")}"
                )
            val namesNel = (rounds.groupBy(_.name).view.mapValues(_.size).filter(_._2 > 1).toList match {
                case Nil => ().asRight
                case namesHisto => s"Repeated name(s) in imaging round sequence! ${namesHisto}".asLeft
            }).toValidatedNel
            (timesNel, namesNel).tupled.toEither.map(_ => ImagingSequence(rounds))
        }
end ImagingSequence