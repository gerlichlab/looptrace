package at.ac.oeaw.imba.gerlich.looptrace

import scala.collection.immutable.SortedSet
import scala.util.Try
import cats.*
import cats.data.NonEmptyList
import cats.syntax.all.*
import io.github.iltotore.iron.:|
import io.github.iltotore.iron.constraint.any.Not
import io.github.iltotore.iron.constraint.numeric.Negative
import mouse.boolean.*
import scopt.OParser
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingTimepoint
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.refinement.IllegalRefinement

import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Combine imaging subfolders to create a single timecourse.
  *
  * https://github.com/gerlichlab/looptrace/issues/137
  */
object CombineImagingFolders extends ScoptCliReaders with StrictLogging:
  val ProgramName = "CombineImagingFolders"

  final case class CliConfig(
      folders: Seq[os.Path] = null, // required
      targetFolder: os.Path = null, // required
      script: os.Path = null, // required
      ext: String = "nd2", // We most commonly store images as ND2.
      execute: Boolean =
        false // By default, just produce a script, don't execute.
  )

  val parserBuilder = OParser.builder[CliConfig]

  def main(args: Array[String]): Unit = {
    import parserBuilder.*

    val parser = OParser.sequence(
      programName(ProgramName),
      head(ProgramName, BuildInfo.version),
      opt[Seq[os.Path]]("folders")
        .required()
        .action((fs, c) => c.copy(folders = fs))
        .validate(fs =>
          fs.filterNot(os.isDir) match {
            case Nil => ().asRight
            case missing =>
              s"${missing.length} missing input folders: $missing".asLeft
          }
        )
        .text("Paths to folders to combine, comma-separated in desired order"),
      opt[os.Path]('O', "targetFolder")
        .required()
        .action((p, c) => c.copy(targetFolder = p))
        .text("Path to folder in which to place output"),
      opt[os.Path]('S', "script")
        .required()
        .action((p, c) => c.copy(script = p))
        .validate(p =>
          (!os.exists(p)).either(s"Script path already exists! $p", ())
        )
        .text("Path to script file to write"),
      opt[String]("ext")
        .action((x, c) => c.copy(ext = x))
        .validate(x =>
          (!x.startsWith("."))
            .either(s"Use pure extension, not period-prefixed.", ())
        )
        .text("Extension for files of interest"),
      opt[Unit]("execute")
        .action((_, c) => c.copy(execute = true))
        .text("Indicate to execute the moves")
    )

    OParser.parse(parser, args, CliConfig()) match {
      case None =>
        throw new Exception(
          s"Illegal CLI use of '${ProgramName}' program. Check --help"
        ) // CLI parser gives error message.
      case Some(opts) =>
        workflow(
          inputFolders = opts.folders,
          filenameFieldSep = "_",
          extToUse = opts.ext,
          script = opts.script,
          targetFolder = opts.targetFolder,
          execute = opts.execute
        )
    }
  }

  def workflow(
      inputFolders: Iterable[os.Path],
      filenameFieldSep: String,
      extToUse: Extension,
      script: os.Path,
      targetFolder: os.Path,
      execute: Boolean
  ): Unit = {
    val infolders =
      if (inputFolders.size < 2) then
        throw new IllegalArgumentException("Need at least 2 input folders!")
      else inputFolders.toList.toNel.get
    prepareUpdatedTimepoints(
      infolders,
      extToUse = extToUse,
      filenameFieldSep = filenameFieldSep,
      targetFolder = targetFolder
    ) flatMap { updates =>
      val (errors, srcDstPairs) =
        if updates.isEmpty then List() -> List()
        else
          Alternative[List].separate(
            updates.toList.map(
              makeSrcDstPair(targetFolder, filenameFieldSep).tupled
            )
          )
      errors.toNel.toLeft(srcDstPairs)
    } match {
      case Left(errors) =>
        val numErrsPreview = 3
        throw new Exception(
          s"${errors.length} errors! First $numErrsPreview max: ${errors.take(numErrsPreview)}"
        )
      case Right(pairs) =>
        // TODO: handle case in which output folder doesn't yet exist.
        checkSrcDstPairs(pairs)
        logger.info(s"Writing script: $script")
        os.write(script, pairs.map((src, dst) => s"mv '$src' '$dst'\n"))
        if execute then {
          logger.info(s"Executing ${pairs.length} moves")
          pairs.foreach(os.move(_, _))
        } else {
          logger.info("Skipping execution")
        }
        logger.info("Done!")
    }
  }

  def checkSrcDstPairs(pairs: List[(os.Path, os.Path)]): Unit = {
    val (srcs, dsts) = pairs.unzip
    def getReps(name: String, paths: List[os.Path]) = {
      val reps =
        paths.groupBy(identity).view.mapValues(_.length).filter(_._2 > 1)
      reps.isEmpty.either(s"Repeats in src $name list: $reps", ())
    }
    val crossover = srcs.toSet & dsts.toSet
    val (errors, _) = Alternative[List].separate(
      List(
        getReps("src", srcs),
        getReps("dst", dsts),
        crossover.isEmpty.either(s"src-dst crossover: $crossover", ())
      )
    )
    if errors.nonEmpty then
      throw new Exception(
        s"${errors.length} error(s) validating move pairs: $errors"
      )
  }

  def makeSrcDstPair(targetFolder: os.Path, sep: String)(
      newTime: ImagingTimepoint,
      oldPath: os.Path
  ): Either[UnusableTimepointUpdateException, (os.Path, os.Path)] = {
    val oldFields = oldPath.last.split(sep)
    ImagingTimepoint
      .parseValueIndexPairFromPath(oldPath, filenameFieldSep = sep)
      .bimap(
        UnusableTimepointUpdateException(oldPath, newTime, _),
        (oldTime, i) =>
          val (preFields, postFields) = oldFields.splitAt(i)
          val newFields: Array[String] = preFields ++ Array(
            ImagingTimepoint.printForFilename(newTime)
          ) ++ postFields.tail
          val fn = newFields `mkString` sep
          oldPath -> (targetFolder / fn)
      )
  }

  def prepareUpdatedTimepoints(
      inputFolders: NonEmptyList[os.Path],
      extToUse: Extension,
      filenameFieldSep: String,
      targetFolder: os.Path
  ): Either[
    NonEmptyList[UnparseablePathException] |
      NonEmptyList[UnusableSubfolderException],
    List[(ImagingTimepoint, os.Path)]
  ] = {
    val keepFile = (_: os.Path).ext === extToUse
    Alternative[List]
      .separate(inputFolders.toList.map { p =>
        prepareSubfolder(keepFile)(p, filenameFieldSep).map(p -> _)
      })
      .bimap(
        _.toNel.map(_.flatten),
        folderContentPairs =>
          val (unusables, prepped) = Alternative[List].separate(
            folderContentPairs.map { (subfolder, contents) =>
              ensureUniqueTimepointsContiguousFromZero(contents.map(_._1))
                .bimap(
                  UnparseablePathException(subfolder, _),
                  _ -> contents
                )
            }
          )
          unusables.toNel.toLeft(prepped)
      ) match {
      case (Some(unusableSubfolderErrors), _) => unusableSubfolderErrors.asLeft
      case (None, preppedSubfolders) =>
        preppedSubfolders.map {
          case Nil                => Nil
          case (_, result) :: Nil => result
          case subs @ ((n1, first) :: rest) =>
            rest
              .foldLeft(n1 -> first.toVector) {
                case ((accCount, timePathPairs), (n, sub)) =>
                  val newCount =
                    val raw = accCount + n
                    NonnegativeInt
                      .option(raw)
                      .getOrElse(
                        throw IllegalRefinement(
                          raw,
                          "Raw accumulated count cannot be refine as nonnegative"
                        )
                      )
                  // TODO: use the ImagingTimepoint.shift functionality.
                  val newPairs =
                    timePathPairs ++ sub.map(_.leftMap(_.unsafeShift(accCount)))
                  newCount -> newPairs
              }
              ._2
              .toList
        }
    }
  }

  /** Ensure subfolder contents are nonempty, and that the timepoints are
    * continuous starting from 0.
    *
    * @param subfolderContents
    *   Collection of pairs of timepoint and path from which it was parsed, all
    *   from the same subfolder
    * @return
    *   Either an error message or the number of unique timepoints
    */
  def ensureUniqueTimepointsContiguousFromZero(
      timepoints: List[ImagingTimepoint]
  ): Either[String, Int :| Not[Negative]] =
    timepoints.toNel
      .toRight("Empty set of timepoints!")
      .flatMap { ts =>
        val repeats =
          ts.groupBy(identity).view.mapValues(_.size).filter(_._2 > 1)
        repeats.isEmpty.either(
          s"${repeats.size} repeated timepoints: ${repeats}",
          ts.toNes
        )
      }
      .flatMap { ts =>
        val numTs = ts.size.toInt
        val ideal = NonEmptyList(0, (1 until numTs).toList)
          .map(ImagingTimepoint.unsafe)
          .toNes
        (ts === ideal).either(
          "Timepoints don't form contiguous sequence from 0.",
          NonnegativeInt
            .option(numTs)
            .getOrElse(
              throw IllegalRefinement(
                numTs,
                s"Number of timepoints ($numTs) can't be refined as nonnegative"
              )
            )
        )
      }

  /** Select files to use, and map to {@code Right}-wrapped collection of pairs
    * of time and path, or {@code Left}-wrapped errors.
    */
  def prepareSubfolder(
      keepFile: os.Path => Boolean
  )(folder: os.Path, filenameFieldSep: String): Either[NonEmptyList[
    UnparseablePathException
  ], List[(ImagingTimepoint, os.Path)]] = {
    val (bads, goods) =
      Alternative[List].separate(
        os.list(folder)
          .toList
          .filter(f => os.isFile(f) && keepFile(f))
          .map { p =>
            ImagingTimepoint
              .parseValueIndexPairFromPath(p, filenameFieldSep)
              .bimap(UnparseablePathException(p, _), _._1 -> p)
          }
      )
    bads.toNel.toLeft(goods)
  }

  private type Extension = String

  final case class UnparseablePathException(path: os.Path, message: String)
      extends Exception(s"$path: $message")

  final case class UnusableSubfolderException(path: os.Path, message: String)
      extends Exception(s"$path: $message")

  final case class UnusableTimepointUpdateException(
      path: os.Path,
      time: ImagingTimepoint,
      message: String
  ) extends Exception(s"($path, $time): $message")
end CombineImagingFolders
