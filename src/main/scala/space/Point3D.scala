package at.ac.oeaw.imba.gerlich.looptrace.space

import cats.data.{ NonEmptyList as NEL }
import upickle.default.*

final case class Point3D(x: XCoordinate, y: YCoordinate, z: ZCoordinate)

enum CoordinateSequence derives ReadWriter:
    case Forward, Reverse

object Point3D {
    import CoordinateSequence.*

    def toList(coordseq: CoordinateSequence)(pt: Point3D): NEL[Coordinate] = {
        val (head, tail) = coordseq match {
            case Forward => (pt.x, pt.z)
            case Reverse => (pt.z, pt.x)
        }
        NEL(head, List(pt.y, tail))
    }

}
