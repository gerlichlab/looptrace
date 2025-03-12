package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{ NotGiven, Try }
import cats.*
import cats.data.*
import cats.syntax.all.*
import mouse.boolean.*
import squants.MetricSystem
import squants.space.{ Length, LengthUnit, Nanometers }
import upickle.core.Abort
import upickle.default.*

import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.geometry
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.coordinate.given
import at.ac.oeaw.imba.gerlich.gerlib.json.JsonValueWriter
import at.ac.oeaw.imba.gerlich.gerlib.json.instances.geometry.getPlainJsonValueWriter
import at.ac.oeaw.imba.gerlich.gerlib.json.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.json.syntax.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

/** Working with (3D) space */
package object space:

    private type Wrapped = Double
    private[looptrace] type RawCoordinate = Wrapped


    type Coordinate = geometry.Coordinate[Wrapped]

    object Coordinate:
        /** Get a writer of coordinate components as JSON values */
        def getJsonWriter[C <: Coordinate: [C] =>> NotGiven[C =:= Coordinate]]: JsonValueWriter[C, ujson.Num] = 
            getPlainJsonValueWriter[Double, C, ujson.Num]

    type XCoordinate = geometry.XCoordinate[Wrapped]

    object XCoordinate:
        def apply(x: Double): XCoordinate = geometry.XCoordinate(x)

    type YCoordinate = geometry.YCoordinate[Wrapped]

    object YCoordinate:
        def apply(y: Double): YCoordinate = geometry.YCoordinate(y)

    type ZCoordinate = geometry.ZCoordinate[Wrapped]

    object ZCoordinate:
        def apply(z: Double): ZCoordinate = geometry.ZCoordinate(z)

    /** Designate the ordering of coordinates for a point (e.g. xyz for Forward or zyx for Reverse). */
    enum CoordinateSequence derives ReadWriter:
        case Forward, Reverse

    object CoordinateSequence:
        /** Conversion of a coordinate sequence member value to JSON value, routing through text representation */
        given (liftStr: String => ujson.Value) => (CoordinateSequence => ujson.Value) = cs => liftStr(cs.toString)

        /** Try to parse given text as enum instance. */
        def parse(s: String): Either[Throwable, CoordinateSequence] = Try{ CoordinateSequence.valueOf(s) }.toEither

        /** Try to parse given JSON value as enum instance. */
        def fromJsonSafe(json: ujson.Value): Either[Throwable, CoordinateSequence] = Try(json.str).toEither.flatMap(parse)

    type BoundingBox = geometry.BoundingBox[Wrapped]

    type Point3D = geometry.Point3D[Wrapped]

    /** Helpers for working with points in 3D space */
    object Point3D:

        def apply(x: XCoordinate, y: YCoordinate, z: ZCoordinate): Point3D = 
            geometry.Point3D(x = x, y = y, z = z)

        /** Order component-wise, (z, y, x) */
        given orderForPoint3D: Order[Point3D] = Order.by(pt => (pt.z, pt.y, pt.x))

        private val jsonKeyX = "xc"
        private val jsonKeyY = "yc"
        private val jsonKeyZ = "zc"

        given JsonValueWriter[Point3D, ujson.Obj]:
            override def apply(p: Point3D): ujson.Obj = 
                import at.ac.oeaw.imba.gerlich.gerlib.json
                ujson.Obj(
                    jsonKeyX -> Coordinate.getJsonWriter[XCoordinate](p.x), 
                    jsonKeyY -> Coordinate.getJsonWriter[YCoordinate](p.y),
                    jsonKeyZ -> Coordinate.getJsonWriter[ZCoordinate](p.z),
                )

        given upickle.default.Reader[Point3D] = 
            summon[upickle.default.Reader[Map[String, Double]]].map(
                _.toList.sortBy(_._1) match {
                    case (`jsonKeyX`, x) :: (`jsonKeyY`, y) :: (`jsonKeyZ`, z) :: Nil => 
                        Point3D(XCoordinate(x), YCoordinate(y), ZCoordinate(z))
                    case kvs => 
                        val msg = s"Unexpected collection of keys for reading point from JSON: ${kvs.map(_._1).mkString("; ")}"
                        throw new Abort(msg)
                }
            )
    
    /** Length in nanometers, restricted to be nonnegative */
    opaque type LengthInNanometers = NonnegativeReal

    /** Helpers for working with length in nanometers */
    object LengthInNanometers:
        def parseString(s: String): EitherNel[String, LengthInNanometers] = 
            given Eq[LengthUnit] = Eq.fromUniversalEquals
            Length.parseString(s)
                .toEither
                .leftMap(e => NonEmptyList.one(e.getMessage))
                .flatMap{ l => 
                    val checkUnitNel = (l.unit === Nanometers).validatedNel(
                        s"Parsed unit (from $s) isn't nanometers, but ${l.unit}", 
                        l.value
                    )
                    val refinedNel = NonnegativeReal.either(l.value)
                        .leftMap{ msg =>
                            s"Can't build nanometers length value (from $s): $msg"
                        }
                        .toValidatedNel
                    (checkUnitNel, refinedNel).mapN((_, l) => l).toEither
                }

        def fromSquants(l: Length): Either[String, LengthInNanometers] = 
            NonnegativeReal.either((l in Nanometers).value)
                .leftMap(msg => s"Error converting length ($l) to (nonnegative) nanometers: $msg")

        def unsafeFromSquants(l: Length): LengthInNanometers = 
            fromSquants(l).fold(msg => throw new Exception(msg), identity)

        given orderForLengthInNanometers: Order[LengthInNanometers] = 
            import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given
            summon[Order[NonnegativeReal]].contramap(l => l: NonnegativeReal)

        given (showX: SimpleShow[NonnegativeReal]) => SimpleShow[LengthInNanometers] = 
            showX.contramap(l => l: NonnegativeReal)

        given LengthLike[LengthInNanometers] = LengthLike.instance(l => Nanometers(l.toDouble))
    end LengthInNanometers

    trait LengthLike[L]:
        def toSquants: L => Length
    end LengthLike

    object LengthLike:
        def instance[A](f: A => Length): LengthLike[A] = new:
            override def toSquants: A => Length = f
        
        object syntax:
            extension [L](l: L)
                def toSquants(using ll: LengthLike[L]): Length = ll.toSquants(l)
    end LengthLike

    // TODO: try to restrict the .symbol abstract member to be "px" singleton.
    opaque type PixelDefinition = LengthUnit

    /** A fundamental unit of length in imaging, the pixel */
    object PixelDefinition:
        /** Define a unit of length in pixels by specifying number of nanometers per pixel. */
        def tryToDefine(onePixelIs: Length): Either[String, PixelDefinition] = 
            PositiveReal
                .either((onePixelIs in Nanometers).value)
                .bimap(msg => s"Cannot define pixel by given length ($onePixelIs): $msg", defineByNanometers)

        def tryToDefine(onePixelIs: LengthInNanometers): Either[String, PixelDefinition] = 
            import LengthLike.syntax.toSquants
            import LengthInNanometers.given
            tryToDefine(onePixelIs.toSquants)
        
        def unsafeDefine(onePixelIs: Length): PixelDefinition = 
            tryToDefine(onePixelIs)
                .leftMap{ msg => new Exception(msg) }
                .fold(throw _, identity)
        
        given Show[PixelDefinition] = 
            Show.show(pxDef => s"PixelDefinition: ${pxDef(1)}")

        /** Define a unit of length in pixels by specifying number of nanometers per pixel. */
        private def defineByNanometers(nmPerPx: PositiveReal): PixelDefinition = new:
            val conversionFactor: Double = nmPerPx * MetricSystem.Nano
            val symbol: String = "px"

        object syntax:
            extension (pxDef: PixelDefinition)
                def lift[A: Numeric](a: A): Length = (pxDef: LengthUnit).apply(a)
    end PixelDefinition

    /** Rescaling of the units in 3D */
    final case class Pixels3D(
        private val x: PixelDefinition, 
        private val y: PixelDefinition, 
        private val z: PixelDefinition,
    ):
        import PixelDefinition.syntax.lift
        def liftX[A: Numeric](a: A): Length = x.lift(a)
        def liftY[A: Numeric](a: A): Length = y.lift(a)
        def liftZ[A: Numeric](a: A): Length = z.lift(a)
    end Pixels3D

end space