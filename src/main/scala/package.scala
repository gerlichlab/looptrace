package at.ac.oeaw.imba.gerlich

import java.io.File
import scala.util.Try
import cats.{ Eq, Show }
import cats.syntax.eq.*
import cats.syntax.show.*

import mouse.boolean.*
import scopt.Read

/** Chromatin fiber tracing with FISH probes */
package object looptrace {
    val VersionName = "0.3.0-SNAPSHOT"

    /** Get the labels of a Product. */
    inline def labelsOf[A](using p: scala.deriving.Mirror.ProductOf[A]) = scala.compiletime.constValueTuple[p.MirroredElemLabels]

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
        def either(z: Int): Either[String, NonnegativeInt] = maybe(z).toRight(s"Cannot refine as nonnegative: $z")
        def indexed[A](xs: List[A]): List[(A, NonnegativeInt)] = {
            // guaranteed nonnegative by construction here
            xs.zipWithIndex.map{ case (x, i) => x -> unsafe(i) }
        }
        def maybe(z: Int): Option[NonnegativeInt] = (z >= 0).option((z: NonnegativeInt))
        def unsafe(z: Int): NonnegativeInt = either(z).fold(msg => throw new NumberFormatException(msg), identity)
        given nonnegativeIntEq: Eq[NonnegativeInt] = Eq.fromUniversalEquals[NonnegativeInt]

    /** Refinement type for nonnegative integers */
    opaque type PositiveInt <: Int = Int
    
    /** Helpers for working with nonnegative integers */
    object PositiveInt:
        inline def apply(z: Int): PositiveInt = 
            inline if z <= 0 then compiletime.error("Non-positive integer where positive is required!")
            else (z: PositiveInt)
        def either(z: Int): Either[String, PositiveInt] = maybe(z).toRight(s"Cannot refine as positive: $z")
        def maybe(z: Int): Option[PositiveInt] = (z > 0).option((z: PositiveInt))
        def unsafe(z: Int): PositiveInt = either(z).fold(msg => throw new NumberFormatException(msg), identity)
        given posIntEq: Eq[PositiveInt] = Eq.fromUniversalEquals[PositiveInt]
        extension (n: PositiveInt)
            def asNonnegative: NonnegativeInt = NonnegativeInt.unsafe(n)
    

    enum Delimiter(val sep: String, val ext: String):
        case CommaSeparator extends Delimiter(",", "csv")
        case TabSeparator extends Delimiter("\t", "tsv")

        def canonicalExtension: String = ext
        def join(fields: Array[String]): String = fields mkString sep
        def split(s: String): Array[String] = split(s, -1)
        def split(s: String, limit: Int): Array[String] = s.split(sep, limit)
    

    object Delimiter:
        def fromPath(p: os.Path): Option[Delimiter] = fromExtension(p.ext)
        def fromExtension(ext: String): Option[Delimiter] = Delimiter.values.filter(_.ext === ext).headOption
    
    
    final case class FrameIndex(get: NonnegativeInt) extends AnyVal
    object FrameIndex:
        implicit val showForFrameIndex: Show[FrameIndex] = Show.show(_.get.show)
    
    final case class PositionIndex(get: NonnegativeInt) extends AnyVal
    object PositionIndex:
        implicit val showForPositionIndex: Show[PositionIndex] = Show.show(_.get.show)

    final case class RoiIndex(get: NonnegativeInt) extends AnyVal
    object RoiIndex:
        implicit val showForRoiIndex: Show[RoiIndex] = Show.show(_.get.show)

    /**
      * Write a mapping, from position and frame pair to value, to JSON.
      *
      * @param vKey The key to use for the {@code V} element in each object
      * @param pfToV The mapping of data to write
      * @param writeV How to write each {@code V} element as JSON
      * @return A JSON array of object corresponding to each element of the map
      */
    def posFrameMapToJson[V](vKey: String, pfToV: Map[(PositionIndex, FrameIndex), V])(using writeV: (V) => ujson.Value): ujson.Value = {
        val proc1 = (pf: (PositionIndex, FrameIndex), v: V) => ujson.Obj(
            "position" -> pf._1.get,
            "frame" -> pf._2.get,
            vKey -> writeV(v)
        )
        pfToV.toList.map(proc1.tupled)
    }
}
