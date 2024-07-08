package at.ac.oeaw.imba.gerlich.looptrace
package syntax

import scala.util.Try
import cats.*
import cats.derived.*
import cats.syntax.all.*
import mouse.boolean.*

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

/** Helpers for working with [[at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint]] values */
trait SyntaxForImagingTimepoint:
    /** The text prefix before the encoding of the numeric timepoint value in a filename */
    private val PrefixInFilename = "Time"
    
    extension (IT: ImagingTimepoint.type)
        /** Attempt to parse a timepoint value from the fields in name of given filepath. */
        def parseValueIndexPairFromPath(p: os.Path, filenameFieldSep: String): Either[String, (ImagingTimepoint, Int)] =
            p.last.split(filenameFieldSep).toList
                .zipWithIndex
                .flatMap{ (s, idx) => parseFilenameField(s).toOption.map(_ -> idx) } match {
                    case pair :: Nil => pair.asRight
                    case times => s"${times.length} timepoints detected from path ($p): $times".asLeft
                }

        /** Parse timepoint from text (typically, a chunk of a delimited filename). */
        private def parseFilenameField(s: String): Either[String, ImagingTimepoint] = s.startsWith(PrefixInFilename)
            .either(s"Timepoint parse input lacks correct prefix ($PrefixInFilename): $s", s.stripPrefix(PrefixInFilename))
                // Read first to Double and then to Int, to ensure no decimal gets through via truncation.
                >>= { (s: String) => Try(s.toDouble).toEither.leftMap(_.getMessage) }
                >>= tryToInt
                >>= IT.fromInt

        /** Like a {@code Show}, but to display the value for use in a filename. */
        def printForFilename(t: ImagingTimepoint): String = PrefixInFilename ++ "%05d".format(t.get)

        /** Assume given integer is nonnegative and lift it into the type for timepoints. */
        def unsafe = IT.fromInt(_: Int).fold(msg => throw new NumberFormatException(msg), identity)
