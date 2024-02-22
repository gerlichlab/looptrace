package at.ac.oeaw.imba.gerlich.looptrace

import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

/** Automatically set the minimum number of passing examples to 100, rather than default of 10, for property-based tests. */
trait ScalacheckSuite extends ScalaCheckPropertyChecks:
    /** Set the default number of successes for a test pass to match Scalacheck. */
    implicit override val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)
