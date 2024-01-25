package at.ac.oeaw.imba.gerlich.looptrace

import cats.data.*
import cats.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.*

/**
 * Behaviors shared by pairwise distance programs
 * 
 * @author Vince Reuter
 */
private[looptrace] trait PairwiseDistanceProgram:
    protected def getColParser[A](header: Array[String])(col: String, lift: String => Either[String, A]): ValidatedNel[String, Array[String] => ValidatedNel[String, A]] =
        header.zipWithIndex
            .find(_._1 === col)
            .map((_, i) => safeGetFromRow(i, lift)(_: Array[String]))
            .toRight(col)
            .toValidatedNel

    protected def preparse(infile: os.Path): (Array[String], List[Array[String]]) = 
        os.read.lines(infile)
            .map(Delimiter.CommaSeparator.split)
            .toList
            .toNel
            .fold(throw EmptyFileException(infile))(recs => recs.head -> recs.tail)

    /** Error type for when a file to use is unexpectedly empty. */
    final case class EmptyFileException(getFile: os.Path) extends Exception(s"File is empty: $getFile")
end PairwiseDistanceProgram
