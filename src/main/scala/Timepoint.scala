package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Try
import cats.*
import cats.syntax.all.*
import mouse.boolean.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.*


final case class Timepoint(get: NonnegativeInt) extends AnyVal

object Timepoint:
    given orderForTimepoint: Order[Timepoint] = Order.by(_.get)
    private val Prefix = "Time"
    
    given showForTimepoint: Show[Timepoint] = Show.show(_.get.show)
    
    def fromInt = NonnegativeInt.either >> Timepoint.apply
    
    def parse(fn: String, filenameFieldSep: String): Either[String, (Timepoint, Int)] = {
        val fields = fn.split(filenameFieldSep)
        fields.zipWithIndex.toList.flatMap{ (s, idx) => parse(s).toOption.map(_ -> idx) } match {
            case pair :: Nil => pair.asRight
            case times => s"${times.length} timepoints detected from filename ($fn): $times".asLeft
        }
    }

    /** Parse timepoint from text (typically, a chunk of a delimited filename). */
    def parse(s: String): Either[String, Timepoint] = 
        // Read first to Double and then to Int, to ensure no decimal gets through via truncation.
        s.startsWith(Prefix).either(s"Timepoint parse input lacks correct prefix ($Prefix): $s", ())
            >>= Function.const{ Try{ s.stripPrefix(Prefix).toDouble }.toEither.leftMap(_.getMessage) }
            >>= tryToInt
            >>= fromInt

    def print(t: Timepoint): String = Prefix ++ "%05d".format(t.get)

    def unsafe = NonnegativeInt.unsafe `andThen` Timepoint.apply
end Timepoint
