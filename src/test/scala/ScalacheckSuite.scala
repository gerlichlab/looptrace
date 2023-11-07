package at.ac.oeaw.imba.gerlich.looptrace

import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

trait ScalacheckSuite extends ScalaCheckPropertyChecks:
    /** Set the default number of successes for a test pass to match Scalacheck. */
    implicit override val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)
