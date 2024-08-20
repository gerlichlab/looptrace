package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Random
import cats.syntax.all.*
import org.scalacheck.Gen
import at.ac.oeaw.imba.gerlich.gerlib.testing.instances.CatsScalacheckInstances

/**
  * Helpers for the [[at.ac.oeaw.imba.gerlich.looptrace.ImagingRound]]-related tests
  * 
  * @author Vince Reuter
  */
trait ImagingRoundHelpers:
    this: CatsScalacheckInstances =>
    def genNameForJson: Gen[String] = (Gen.alphaNumStr, Gen.listOf(Gen.oneOf("_", "-", " ", "."))).mapN(
        (alphaNum, punctuation) => Random.shuffle(alphaNum.toList ::: punctuation.toList).mkString
    ).suchThat(n => n.nonEmpty & !n.forall(_.isWhitespace))
end ImagingRoundHelpers
