package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.{NotGiven, Try}
import cats.*
import cats.data.*
import cats.syntax.all.*
import io.github.iltotore.iron.:|
import io.github.iltotore.iron.cats.given
import io.github.iltotore.iron.constraint.any.Not
import io.github.iltotore.iron.constraint.numeric.{Negative, Positive}
import mouse.boolean.*
import squants.MetricSystem
import squants.space.{Length, LengthUnit, Nanometers}
import upickle.core.Abort
import upickle.default.*

import at.ac.oeaw.imba.gerlich.gerlib.{SimpleShow, geometry}
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
    def getJsonWriter[C <: Coordinate: [C] =>> NotGiven[C =:= Coordinate]]
        : JsonValueWriter[C, ujson.Num] =
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
          jsonKeyZ -> Coordinate.getJsonWriter[ZCoordinate](p.z)
        )

    given upickle.default.Reader[Point3D] =
      summon[upickle.default.Reader[Map[String, Double]]].map(
        _.toList.sortBy(_._1) match {
          case (`jsonKeyX`, x) :: (`jsonKeyY`, y) :: (`jsonKeyZ`, z) :: Nil =>
            Point3D(XCoordinate(x), YCoordinate(y), ZCoordinate(z))
          case kvs =>
            val msg =
              s"Unexpected collection of keys for reading point from JSON: ${kvs.map(_._1).mkString("; ")}"
            throw new Abort(msg)
        }
      )
end space
