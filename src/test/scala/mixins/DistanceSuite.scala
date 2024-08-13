package at.ac.oeaw.imba.gerlich.looptrace

import org.scalacheck.Gen

/** Test suites dealing with computation of distance */
trait DistanceSuite:
    // Prevent overflow in Euclidean distance computation.
    protected def genReasonableCoordinate: Gen[Double] = Gen.choose(-1e16, 1e16)
end DistanceSuite
