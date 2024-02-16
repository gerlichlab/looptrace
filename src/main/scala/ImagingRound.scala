package at.ac.oeaw.imba.gerlich.looptrace

import scala.language.adhocExtensions // for extending ujson.Value.InvalidData
import scala.util.Try
import cats.data.*
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*

import at.ac.oeaw.imba.gerlich.looptrace.syntax.*

/** A round of imaging during an experiment */
sealed trait ImagingRound:
    def name: String
    def time: Timepoint
end ImagingRound

/** Helpers and JSON tools for working with the abstraction of imaging rounds */
object ImagingRound:
    import BlankImagingRound.*
    import LocusImagingRound.*
    import RegionalImagingRound.*

    /**
     * Wrapper around [[ujson.Value.InvalidData]] for the case in which one or more errors occur during decoding.
     * 
     * @param whatWasBeingDecoded Text to provide context to the error message, typically the name of the type of value 
     *    that was being decoded when error(s) wrapped here arose
     * @param json The data that was being decoded
     * @param messages The error messages explaining why decoding didn't work
     */
    final class DecodingError(val whatWasBeingDecoded: String, val json: ujson.Value, val messages: NonEmptyList[String]) 
        extends ujson.Value.InvalidData(json, s"Error(s) decoding ($whatWasBeingDecoded): ${messages.mkString_(", ")}")
    
    /**
     * A JSON read/write implementation for imaging rounds, allowing subtype determination to be done by keys and values present.
     * 
     * Specifically, here are some rules for what determines the subtype that will be decoded:
     * 1. A [[RegionalImagingRound]] value decoding attempt will be made iff the `isRegional` key is present with a value of `true`.
     * 2. A [[BlankImagingRound]] value decoding attempt will be made iff the `isBlank` key is present with a value of `true`.
     * 3. A [[LocusImagingRound]] value decoding attempt will be made otherwise.
     * 4. The presence of `probe` key, giving the name of the FISH probe used for the imaging round, is required whenever the 
     *    `isBlank` key is absent or set to `false`
     * 5. `name` is required exactly when `probe` is not, namely to decode a [[BlankImagingRound]]; this is because otherwise, 
     *    the name of the imaging round will match that of the probe (possibly with the addition of a suffix to encode the `repeat` 
     *    value).
     * 
     * @return A [[upickle.default.ReadWriter]] value with which to read and write values of [[ImagingRound]]
     */
    def rwForImagingRound: ReadWriter[ImagingRound] = readwriter[ujson.Value].bimap(
        roundToJsonObject, 
        json => parseFromJsonMap(json.obj.toMap).fold(errors => throw new DecodingError("ImagingRound", json, errors), identity)
    )

    /** Try to parse a single imaging round instance from a raw JSON data mapping */
    def parseFromJsonMap(data: Map[String, ujson.Value]): ErrMsgsOr[ImagingRound] = {
        val keys = data.keySet
        val timeNel = (data.get("time").toRight("Missing timepoint!") >>= parseTimeValue).toValidatedNel
        val isRegionalNel = extractDefaultFalse("isRegional")(data).toValidatedNel
        val isBlankNel = extractDefaultFalse("isBlank")(data).toValidatedNel
        val nameOptNel = extractOptionalString("name")(data).toValidatedNel
        val probeOptNel = extractOptionalString("probe")(data).toValidatedNel
        val repOptNel = data.get("repeat").traverse(parseRepeatValue).toValidatedNel
        val noExtraKeysNel = validateNoExtraKeys(Set("time", "probe", "name", "isRegional", "isBlank", "repeat"), "ImagingRound")(data)
        (timeNel, isRegionalNel, isBlankNel, nameOptNel, probeOptNel, repOptNel, noExtraKeysNel).tupled.toEither.flatMap{
            case (time, isRegional, isBlank, nameOpt, probeOpt, repOpt, _) => 
                val blankRepNel = (repOpt.isEmpty || !isBlank).either(s"Blank round cannot be a repeat!", ()).toValidatedNel
                val regRepNel = (repOpt.isEmpty || !isRegional).either(s"Regional round cannot be a repeat!", ()).toValidatedNel
                val blankRegionalNel = (!isRegional || !isBlank).either("Image round can't be both blank and regional!", ()).toValidatedNel
                val blankProbeNel = ((probeOpt, isBlank) match {
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
                (blankRepNel, regRepNel, blankRegionalNel, blankProbeNel, nameProbeRepNel).mapN{ 
                    case (_, _, _, _, (name, probe)) => 
                        if isBlank then BlankImagingRound(name, time)
                        else if isRegional then RegionalImagingRound(name, time, probe)
                        else LocusImagingRound(name, time, probe, repOpt)
                }.toEither
        }
    }

    private[looptrace] def roundToJsonObject(imagingRound: ImagingRound): ujson.Obj = imagingRound match {
        case round: LocusImagingRound => locusRoundToJson(round)
        case round: RegionalImagingRound => regionalRoundToJson(round)
        case round: BlankImagingRound => blankRoundToJson(round)
    }

    private def parseTimeValue(v: ujson.Value): Either[String, Timepoint] = 
        Try(v.int).toEither.leftMap(e => s"Non-integral value for time! ${e.getMessage}") 
            >>= NonnegativeInt.either 
            >> Timepoint.apply

    private def parseRepeatValue(v: ujson.Value): Either[String, PositiveInt] = 
        Try(v.int).toEither.leftMap(e => s"Non-integral value for repeat! ${e.getMessage}") 
            >>= PositiveInt.either

    /**
     * Extract a Boolean value from the given key in a key-value mapping to JSON values.
     * 
     * @param key The key of the data element to extract
     * @param data The key-value store from which to extract a datum
     * @return Either a [[scala.util.Left]]-wrapped error message for a bad value present at the given key, 
     *     or a [[scala.util.Right]]-wrapped Boolean value extracted from the given key
     */
    private def extractDefaultFalse(key: String)(data: Map[String, ujson.Value]): Either[String, Boolean] = 
        data.get(key) match {
            case None => false.asRight
            case Some(v) => Try(v.bool).toEither.leftMap(e => s"Illegal value for '$key'! ${e.getMessage}")
        }

    /**
     * Extract a text value from the given key in a key-value mapping to JSON values.
     * 
     * @param key The key of the data element to extract
     * @param data The key-value store from which to extract a datum
     * @return Either a [[scala.util.Left]]-wrapped error message for a bad value present at the given key, 
     *     or a [[scala.util.Right]]-wrapped optional, empty if the key was not present or a nonempty with 
     *     the extracted value if the key was in fact present with a valid value
     */
    private def extractOptionalString(key: String)(data: Map[String, ujson.Value]): Either[String, Option[String]] = 
        data.get(key).traverse{ v => Try(v.str).toEither.leftMap(e => s"Illagel value for '$key'! ${e.getMessage}") }

    /**
     * Check that the given data contains no keys other than those in the set passed as valid.
     * 
     * @param validKeys The set of keys that the given data are permitted to have
     * @param whatToDecode Some sort of name or short phrase to give context to a potential error messages
     * @param data The key-value mapping being checked for presence of any extra keys
     * @return Either a [[cats.data.Validated.Invalid]] wrapping a singleton explanatory message, or a trivial [[cats.data.Validated.Valid]]
     */
    private def validateNoExtraKeys(validKeys: Set[String], whatToDecode: String)(data: Map[String, ujson.Value]): ValidatedNel[String, Unit] = {
        (data.keySet.toSet -- validKeys)
            .toList
            .toNel
            .toLeft(())
            .leftMap(extra => s"Extra key(s) for decoding $whatToDecode: ${extra.mkString_(", ")}")
            .toValidatedNel
    }
end ImagingRound

/**
 * An imaging round in which no hybridisation is done
 * 
 * @param name The name for the imaging round within an experiment
 * @param time The timepoint (0-based, inclusive) of this imaging round within an experiment
 */
final case class BlankImagingRound(name: String, time: Timepoint) extends ImagingRound

object BlankImagingRound:
    private[looptrace] def blankRoundToJson(round: BlankImagingRound) = ujson.Obj(
        "name" -> ujson.Str(round.name), 
        "time" -> ujson.Num(round.time.get), 
        "isBlank" -> ujson.Bool(true),
        )
end BlankImagingRound

/** An imaging round in which FISH is done */
sealed trait FishProbed:
    /** Name for the probe used for hybridisation */
    def probe: ProbeName
    /** Number (1-based, inclusive) of repeat of a particular probe that this round represents within the experiment */
    def repeat: Option[PositiveInt]
end FishProbed

/**
  * Round of imaging in which a specific locus was targeted during experiment
  *
  * @param name Name for the imaging round
  * @param time The timepoint (0-based, inclusive) of this imaging round within an experiment
  * @param probe Name for the probe used for hybridisation
  * @param repeat  Number (1-based, inclusive) of repeat of a particular probe that this round represents within the experiment
  */
final case class LocusImagingRound(name: String, time: Timepoint, probe: ProbeName, repeat: Option[PositiveInt]) extends ImagingRound, FishProbed:
    final def isRepeat = repeat.nonEmpty

/** Helpers and alternate constructors for working with imaging rounds of specific genomic loci */
object LocusImagingRound:
    def apply(time: Timepoint, probe: ProbeName): LocusImagingRound = apply(None, time, probe, None)
    def apply(time: Timepoint, probe: ProbeName, repeat: PositiveInt): LocusImagingRound = apply(None, time, probe, repeat.some)
    def apply(maybeName: Option[String], time: Timepoint, probe: ProbeName, maybeRepeat: Option[PositiveInt]): LocusImagingRound = 
        val name = maybeName.getOrElse(probe.get ++ maybeRepeat.fold("")(rep => s"_repeat${rep.show}"))
        new LocusImagingRound(name, time, probe, maybeRepeat)

    private[looptrace] def locusRoundToJson(round: LocusImagingRound): ujson.Obj = {
        val baseData = NonEmptyList.of(
            "name" -> ujson.Str(round.name), 
            "probe" -> ujson.Str(round.probe.get),
            "time" -> ujson.Num(round.time.get),
            )
        val fullData = baseData ++ round.repeat.fold(List()){ n => List("repeat" -> ujson.Num(n)) }
        ujson.Obj(fullData.head, fullData.tail*)
    }
end LocusImagingRound

/**
  * Round of imaging in which an entire region (multiple individual loci) was targeted during experiment
  *
  * @param name Name for the imaging round
  * @param time The timepoint (0-based, inclusive) of this imaging round within an experiment
  * @param probe Name for the probe used for hybridisation
  */
final case class RegionalImagingRound(name: String, time: Timepoint, probe: ProbeName) extends ImagingRound, FishProbed:
    override final def repeat: Option[PositiveInt] = None
end RegionalImagingRound

/** Helpers and alternate constructors for working with imaging rounds of genomic regions */
object RegionalImagingRound:
    def apply(time: Timepoint, probe: ProbeName): RegionalImagingRound = apply(None, time, probe)
    def apply(maybeName: Option[String], time: Timepoint, probe: ProbeName): RegionalImagingRound = 
        new RegionalImagingRound(maybeName.getOrElse(probe.get), time, probe)

    private[looptrace] def regionalRoundToJson(round: RegionalImagingRound) = ujson.Obj(
        "name" -> round.name, 
        "probe" -> round.probe.get,
        "time" -> round.time.get, 
        "isRegional" -> true
        )
end RegionalImagingRound
