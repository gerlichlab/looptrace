package at.ac.oeaw.imba.gerlich.looptrace.space

sealed trait Coordinate { def get: Double }

final case class XCoordinate(get: Double) extends Coordinate

final case class YCoordinate(get: Double) extends Coordinate

final case class ZCoordinate(get: Double) extends Coordinate
