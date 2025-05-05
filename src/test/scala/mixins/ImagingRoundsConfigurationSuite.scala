package at.ac.oeaw.imba.gerlich.looptrace

import scala.collection.immutable.ArraySeq
import cats.syntax.all.*
import org.scalacheck.*

import at.ac.oeaw.imba.gerlich.gerlib.testing.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.ImagingRoundsConfiguration.*

/** Tools for working with values and types related to the imaging rounds
  * configuration
  */
trait ImagingRoundsConfigurationSuite:
  /** Simply choose one of the enumeration's values/members. */
  given Arbitrary[RoiPartnersRequirementType] = Arbitrary:
    Gen.oneOf(ArraySeq.unsafeWrapArray(RoiPartnersRequirementType.values))

end ImagingRoundsConfigurationSuite
