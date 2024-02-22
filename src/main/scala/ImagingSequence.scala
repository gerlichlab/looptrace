package at.ac.oeaw.imba.gerlich.looptrace

import scala.collection.mutable.{ Map as MMap }
import scala.language.adhocExtensions // for extending ujson.Value.InvalidData
import scala.util.Try
import cats.data.{ NonEmptyList, NonEmptySet, ValidatedNel }
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*

/** A sequence of FISH and blank imaging rounds, constituting a microscopy experiment */
final case class ImagingSequence private(
    blankRounds: List[BlankImagingRound], 
    locusRounds: List[LocusImagingRound], 
    regionRounds: NonEmptyList[RegionalImagingRound],
    ):
    private var lookup: MMap[String, ImagingRound] = MMap()
    final def get(name: String): Option[ImagingRound] = {
        val findIn = (_: List[ImagingRound]).find(_.name === name).toLeft(())
        lookup.get(name).orElse{
            (for {
                _ <- findIn(locusRounds)
                _ <- findIn(regionRounds.toList)
                _ <- findIn(blankRounds)
            } yield ()).swap.toOption.flatMap(lookup.put(name, _))
        }
    }
    final lazy val allTimepoints: NonEmptySet[Timepoint] = 
        (blankRounds ::: locusRounds).foldRight(regionRounds.map(_.time).toNes){ (r, acc) => acc.add(r.time) }
    final def length: Int = allTimepoints.length
    final def size: Int = length
    final def numberOfRounds: Int = length
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
            val timesNel = 
                if times === (0 until ordByTime.length).toList then ().validNel 
                else s"Ordered timepoints for imaging rounds don't form contiguous sequence from 0 up to length (${times.length}): ${times.mkString(", ")}".invalidNel
            val namesNel = (rounds.groupBy(_.name).view.mapValues(_.size).filter(_._2 > 1).toList match {
                case Nil => ().asRight
                case namesHisto => s"Repeated name(s) in imaging round sequence! ${namesHisto}".asLeft
            }).toValidatedNel
            val roundsPartitionNel: ValidatedNel[String, (List[BlankImagingRound], List[LocusImagingRound], NonEmptyList[RegionalImagingRound])] = 
                rounds.toList.foldRight((List.empty[BlankImagingRound], List.empty[LocusImagingRound], List.empty[RegionalImagingRound])){ 
                    case (r: BlankImagingRound, (blanks, locals, regionals)) => (r :: blanks, locals, regionals)
                    case (r: LocusImagingRound, (blanks, locals, regionals)) => (blanks, r :: locals, regionals)
                    case (r: RegionalImagingRound, (blanks, locals, regionals)) => (blanks, locals, r :: regionals)
                } match {
                    case (blanks, loci, h :: t) => (blanks, loci, NonEmptyList(h, t)).validNel
                    case (_, _, Nil) => "No REGIONAL rounds found!".invalidNel
                }
            (timesNel, namesNel, roundsPartitionNel).mapN((_, _, partedRounds) => ImagingSequence.apply.tupled(partedRounds)).toEither
        }

end ImagingSequence