package at.ac.oeaw.imba.gerlich.looptrace

import scala.util.Random
import cats.syntax.all.*
import org.scalacheck.Gen

/**
  * Helpers for the [[at.ac.oeaw.imba.gerlich.looptrace.ImagingRound]]-related tests
  * 
  * @author Vince Reuter
  */
trait ImagingRoundHelpers:
    this: ScalacheckGenericExtras =>
    def genNameForJson: Gen[String] = (Gen.alphaNumStr, Gen.listOf(Gen.oneOf("_", "-", " ", "."))).mapN(
        (alphaNum, punctuation) => Random.shuffle(alphaNum.toList ::: punctuation.toList).mkString
    ).suchThat(n => n.nonEmpty & !n.forall(_.isWhitespace))
end ImagingRoundHelpers
