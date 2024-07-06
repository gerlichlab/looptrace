package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import cats.*
import cats.derived.*
import cats.syntax.all.*
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import NonnegativeInt.given

import at.ac.oeaw.imba.gerlich.looptrace.syntax.*

/** Wrapper around nonnegative integer to represent timepoint index in sequential FISH experiment */
final case class Timepoint(get: NonnegativeInt) derives Order

/** Helpers for working with {@code Timepoint} values */
object Timepoint:
    given showForTimepoint(using ev: Show[NonnegativeInt]): Show[Timepoint] = ev.contramap(_.get)
    given SimpleShow[Timepoint] = SimpleShow.fromShow
    
    /** The text prefix before the encoding of the numeric timepoint value in a filename */
    private val PrefixInFilename = "Time"
    
    /** Attempt to create a timepoint from an integer, first refining through {@code NonnegativeInt}. */
    def fromInt = NonnegativeInt.either >> Timepoint.apply
    
    /** Attempt to parse a {@code Timepoint} value from the fields in name of given filepath. */
    def parseValueIndexPairFromPath(p: os.Path, filenameFieldSep: String): Either[String, (Timepoint, Int)] =
        p.last.split(filenameFieldSep).toList
            .zipWithIndex
            .flatMap{ (s, idx) => parseFilenameField(s).toOption.map(_ -> idx) } match {
                case pair :: Nil => pair.asRight
                case times => s"${times.length} timepoints detected from path ($p): $times".asLeft
            }

    /** Parse timepoint from text (typically, a chunk of a delimited filename). */
    private def parseFilenameField(s: String): Either[String, Timepoint] = s.startsWith(PrefixInFilename)
        .either(s"Timepoint parse input lacks correct prefix ($PrefixInFilename): $s", s.stripPrefix(PrefixInFilename))
            // Read first to Double and then to Int, to ensure no decimal gets through via truncation.
            >>= { (s: String) => Try(s.toDouble).toEither.leftMap(_.getMessage) }
            >>= tryToInt
            >>= fromInt

    /** Like a {@code Show}, but to display the value for use in a filename. */
    def printForFilename(t: Timepoint): String = PrefixInFilename ++ "%05d".format(t.get)

    /** Assume given integer is nonnegative and lift it into the type for timepoints. */
    def unsafe = fromInt(_: Int).fold(msg => throw new NumberFormatException(msg), identity)
end Timepoint
