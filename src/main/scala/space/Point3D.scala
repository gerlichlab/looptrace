package at.ac.oeaw.imba.gerlich.looptrace.space

import cats.data.{ NonEmptyList as NEL }
import cats.syntax.either.*
import upickle.default.*

final case class Point3D(x: XCoordinate, y: YCoordinate, z: ZCoordinate)

enum CoordinateSequence derives ReadWriter:
    case Forward, Reverse

object Point3D {
    import CoordinateSequence.*

    def fromList(coordseq: CoordinateSequence)(pt: List[Double]): Either[String, Point3D] = {
        pt match {
            case _1 :: _2 :: _3 :: Nil => {
                val y = YCoordinate(_2)
                val (x, z) = coordseq match {
                    case Forward => XCoordinate(_1) -> ZCoordinate(_3)
                    case Reverse => XCoordinate(_3) -> ZCoordinate(_1)
                }
                Point3D(x, y, z).asRight
            }
            case _ => s"Expected 3 coordinates for a 3D point, but got ${pt.length}".asLeft
        }
    }

    def toList(coordseq: CoordinateSequence)(pt: Point3D): NEL[Coordinate] = {
        val (head, tail) = coordseq match {
            case Forward => (pt.x, pt.z)
            case Reverse => (pt.z, pt.x)
        }
        NEL(head, List(pt.y, tail))
    }

}
