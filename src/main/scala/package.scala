package at.ac.oeaw.imba.gerlich

import java.io.File
import cats.{ Eq, Show }
import cats.syntax.show.*
import mouse.boolean.*
import scopt.Read

/** Chromatin fiber tracing with FISH probes */
package object looptrace {
    val VersionName = "0.3.0-SNAPSHOT"

    /** Allow custom types as CLI parameters. */
    object CliReaders:
        given pathRead(using fileRead: Read[File]): Read[os.Path] = fileRead.map(os.Path.apply)
        given nonNegIntRead(using intRead: Read[Int]): Read[NonnegativeInt] = intRead.map(NonnegativeInt.unsafe)
        given posIntRead(using intRead: Read[Int]): Read[PositiveInt] = intRead.map(PositiveInt.unsafe)

    /** Refinement type for nonnegative integers */
    opaque type NonnegativeInt <: Int = Int
    
    /** Helpers for working with nonnegative integers */
    object NonnegativeInt:
        inline def apply(z: Int): NonnegativeInt = 
            inline if z < 0 then compiletime.error("Negative integer where nonnegative is required!")
            else (z: NonnegativeInt)
        def either(z: Int): Either[String, NonnegativeInt] = 
            maybe(z).toRight(s"Cannot refine as nonnegative: $z")
        def indexed[A](xs: List[A]): List[(A, NonnegativeInt)] = {
            // guaranteed nonnegative by construction here
            xs.zipWithIndex.map{ case (x, i) => x -> unsafe(i) }
        }
        def maybe(z: Int): Option[NonnegativeInt] = (z >= 0).option((z: NonnegativeInt))
        def unsafe(z: Int): NonnegativeInt = maybe(z).getOrElse{ throw new NumberFormatException(f"Not nonnegative: $z") }

    /** Refinement type for nonnegative integers */
    opaque type PositiveInt <: Int = Int
    
    /** Helpers for working with nonnegative integers */
    object PositiveInt:
        inline def apply(z: Int): PositiveInt = 
            inline if z <= 0 then compiletime.error("Non-positive integer where positive is required!")
            else (z: PositiveInt)
        def either(z: Int): Either[String, PositiveInt] = 
            maybe(z).toRight(s"Cannot refine as positive: $z")
        def maybe(z: Int): Option[PositiveInt] = (z > 0).option((z: PositiveInt))
        def unsafe(z: Int): PositiveInt = maybe(z).getOrElse{ throw new NumberFormatException(f"Not positive: $z") }
                
    sealed trait Delimiter {
        def canonicalExtension: String
        def show: String
        final def join(fields: Array[String]): String = fields mkString show
        infix def split(s: String): Array[String] = split(s, 0)
        def split(s: String, limit: Int): Array[String] = s.split(show, -1)
    }

    object Delimiter:
        def infer(p: os.Path): Option[Delimiter] = Map("csv" -> CommaSeparator, "tsv" -> TabSeparator).get(p.ext)

    case object TabSeparator extends Delimiter {
        override def canonicalExtension: String = "tsv"
        override def show: String = "\t"
    }

    case object CommaSeparator extends Delimiter {
        override def canonicalExtension: String = "csv"
        override def show: String = ","
    }

    final case class FrameIndex(get: NonnegativeInt) extends AnyVal
    
    object FrameIndex:
        implicit val showForFrameIndex: Show[FrameIndex] = Show.show(_.get.show)
    
    final case class PositionIndex(get: NonnegativeInt) extends AnyVal

    object PositionIndex:
        implicit val showForPositionIndex: Show[PositionIndex] = Show.show(_.get.show)

    final case class RoiIndex(get: NonnegativeInt) extends AnyVal

    object RoiIndex:
        implicit val showForRoiIndex: Show[RoiIndex] = Show.show(_.get.show)

}
