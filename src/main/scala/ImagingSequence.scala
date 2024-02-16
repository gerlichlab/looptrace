package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import upickle.default.*

import cats.data.*
import cats.syntax.all.*
import mouse.boolean.*

/** A sequence of FISH and blank imaging rounds, constituting a microscopy experiment */
final case class ImagingSequence private(rounds: NonEmptyList[ImagingRound]):
    lazy val _lookup = rounds.map(r => r.name -> r).toNem
    def get(name: String): Option[ImagingRound] = _lookup(name)
end ImagingSequence

/** Smart constructors and tools for working with sequences of imaging rounds */
object ImagingSequence:
    private def preprocess1(data: ujson.Value): Either[String, Map[String, ujson.Value]] = 
        Try(data.obj).toEither.bimap(_.getMessage, _.toMap)
    
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