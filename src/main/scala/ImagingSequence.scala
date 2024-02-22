package at.ac.oeaw.imba.gerlich.looptrace

import scala.language.adhocExtensions // for extending ujson.Value.InvalidData
import scala.util.Try
import cats.data.NonEmptyList
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*

/** A sequence of FISH and blank imaging rounds, constituting a microscopy experiment */
final case class ImagingSequence private(rounds: NonEmptyList[ImagingRound]):
    lazy val _lookup = rounds.map(r => r.name -> r).toNem
    final def get(name: String): Option[ImagingRound] = _lookup(name)
    final def length: Int = rounds.length
    final def size: Int = length
    final def numberOfRounds: Int = size
    final def getLocusRounds: List[LocusImagingRound] = rounds.toList.flatMap(ImagingRound.toLocal)
    final def getRegionRounds: List[RegionalImagingRound] = rounds.toList.flatMap(ImagingRound.toRegional)
end ImagingSequence

/** Smart constructors and tools for working with sequences of imaging rounds */
object ImagingSequence:
    
    /** When JSON can't be decoded as sequence of imaging rounds */
    class DecodingError(messages: NonEmptyList[String], json: ujson.Value) 
        extends ujson.Value.InvalidData(json, s"Error(s) decoding ImagingSequence: ${messages.mkString_(", ")}")

    /**
      * Create a sequence of imaging rounds constituting an imaging experiment.
      * 
      * With the individual rounds already parsed, the additional validation / error cases here 
      * are with respect to the non-emptiness of the collection, the sequence of timepoints, and 
      * the uniqueness of the names of the rounds (assumed to be usable as an identifier within an experiment).
      * 
      * To parse a valid instance, the given collection of individual rounds should be nonempty, 
      * the timepoints should form a sequence '(0, 1, ..., N-1)', with 'N' representing the number 
      * of imaging rounds, and there should be no repeated name among the rounds.
      *
      * @param maybeRounds Sequence of individual imaging rounds
      * @return Either a [[scala.util.Left]]-wrapped [[cats.data.NonEmptyList]] of error messages, 
      *     or a [[scala.util.Right]]-wrapped [[ImagingSequence]] instance
      */
    def fromRounds(maybeRounds: List[ImagingRound]): ErrMsgsOr[ImagingSequence] = maybeRounds.toNel
        .toRight(NonEmptyList.one("Can't build an imaging sequence from empty collection of rounds!"))
        .flatMap{ rounds => 
            val ordByTime = rounds.sortBy(_.time)
            val times = ordByTime.map(_.time.get.toInt).toList
            val timesNel = (times === (0 until ordByTime.length).toList).either(
                s"Ordered timepoints for imaging rounds don't form contiguous sequence from 0 up to length (${times.length}): ${times.mkString(", ")}", 
                ()).toValidatedNel
            val namesNel = (rounds.groupBy(_.name).view.mapValues(_.size).filter(_._2 > 1).toList match {
                case Nil => ().asRight
                case namesHisto => s"Repeated name(s) in imaging round sequence! ${namesHisto}".asLeft
            }).toValidatedNel
            (timesNel, namesNel).tupled.toEither.map(_ => ImagingSequence(rounds))
        }

end ImagingSequence