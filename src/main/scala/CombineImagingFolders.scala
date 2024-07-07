package at.ac.oeaw.imba.gerlich.looptrace

import scala.collection.immutable.SortedSet
import scala.util.Try
import cats.*
import cats.data.{ NonEmptyList as NEL }
import cats.syntax.all.*
import mouse.boolean.*
import scopt.OParser
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.numeric.*

import at.ac.oeaw.imba.gerlich.looptrace.syntax.*

/** Combine imaging subfolders to create a single timecourse.
 * 
 * https://github.com/gerlichlab/looptrace/issues/137 
 */
object CombineImagingFolders extends StrictLogging:
    val ProgramName = "CombineImagingFolders"

    final case class CliConfig(
        folders: Seq[os.Path] = null,     // required
        targetFolder: os.Path = null,     // required
        script: os.Path = null,           // required
        ext: String = "nd2",              // We most commonly store images as ND2.
        execute: Boolean = false,         // By default, just produce a script, don't execute.
        )

    val parserBuilder = OParser.builder[CliConfig]
    
    def main(args: Array[String]): Unit = {
        import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders.given
        import parserBuilder.*

        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, VersionName), 
            opt[Seq[os.Path]]("folders")
                .required()
                .action((fs, c) => c.copy(folders = fs))
                .validate(fs => fs.filterNot(os.isDir) match {
                    case Nil => ().asRight
                    case missing => s"${missing.length} missing input folders: $missing".asLeft
                })
                .text("Paths to folders to combine, comma-separated in desired order"), 
            opt[os.Path]('O', "targetFolder")
                .required()
                .action((p, c) => c.copy(targetFolder = p))
                .text("Path to folder in which to place output"),
            opt[os.Path]('S', "script")
                .required()
                .action((p, c) => c.copy(script = p))
                .validate(p => (!os.exists(p)).either(s"Script path already exists! $p", ()))
                .text("Path to script file to write"),
            opt[String]("ext")
                .action((x, c) => c.copy(ext = x))
                .validate(x => (!x.startsWith(".")).either(s"Use pure extension, not period-prefixed.", ()))
                .text("Extension for files of interest"),
            opt[Unit]("execute")
                .action((_, c) => c.copy(execute = true))
                .text("Indicate to execute the moves")
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception(
                s"Illegal CLI use of '${ProgramName}' program. Check --help"
                ) // CLI parser gives error message.
            case Some(opts) => workflow(
                inputFolders = opts.folders, 
                filenameFieldSep = "_", 
                extToUse = opts.ext, 
                script = opts.script, 
                targetFolder = opts.targetFolder, 
                execute = opts.execute,
                )
        }
    }

    def workflow(inputFolders: Iterable[os.Path], filenameFieldSep: String, extToUse: Extension, script: os.Path, targetFolder: os.Path, execute: Boolean): Unit = {
        val infolders = if (inputFolders.size < 2) then throw new IllegalArgumentException("Need at least 2 input folders!") else inputFolders.toList.toNel.get
        prepareUpdatedTimepoints(
            infolders, 
            extToUse = extToUse, 
            filenameFieldSep = filenameFieldSep, 
            targetFolder = targetFolder
        ) flatMap { updates => 
            val (errors, srcDstPairs) = 
                if updates.isEmpty then List() -> List() 
                else Alternative[List].separate(updates.toList.map(makeSrcDstPair(targetFolder, filenameFieldSep).tupled))
            errors.toNel.toLeft(srcDstPairs)
        } match {
            case Left(errors) => 
                val numErrsPreview = 3
                throw new Exception(s"${errors.length} errors! First $numErrsPreview max: ${errors.take(numErrsPreview)}")
            case Right(pairs) => 
                // TODO: handle case in which output folder doesn't yet exist.
                checkSrcDstPairs(pairs)
                logger.info(s"Writing script: $script")
                os.write(script, pairs.map((src, dst) => s"mv '$src' '$dst'\n"))
                if (execute) {
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
            val reps = paths.groupBy(identity).view.mapValues(_.length).filter(_._2 > 1)
            reps.isEmpty.either(s"Repeats in src $name list: $reps", ())
        }
        val crossover = srcs.toSet & dsts.toSet
        val (errors, _) = Alternative[List].separate(List(
            getReps("src", srcs), 
            getReps("dst", dsts), 
            crossover.isEmpty.either(s"src-dst crossover: $crossover", ())
            ))
        if (errors.nonEmpty) throw new Exception(s"${errors.length} error(s) validating move pairs: $errors")
    }

    def makeSrcDstPair(targetFolder: os.Path, sep: String)(newTime: Timepoint, oldPath: os.Path): Either[UnusableTimepointUpdateException, (os.Path, os.Path)] = {
        val oldFields = oldPath.last.split(sep)
        Timepoint.parseValueIndexPairFromPath(oldPath, filenameFieldSep = sep).bimap(
            UnusableTimepointUpdateException(oldPath, newTime, _),  
            (oldTime, i) => 
                val (preFields, postFields) = oldFields.splitAt(i)
                val newFields: Array[String] = preFields ++ Array(Timepoint.printForFilename(newTime)) ++ postFields.tail
                val fn = newFields `mkString` sep
                oldPath -> (targetFolder / fn)
            )
    }
    
    def prepareUpdatedTimepoints(inputFolders: NEL[os.Path], extToUse: Extension, filenameFieldSep: String, targetFolder: os.Path): 
        Either[NEL[UnparseablePathException] | NEL[UnusableSubfolderException], List[(Timepoint, os.Path)]] = {
        val keepFile = (_: os.Path).ext === extToUse
        Alternative[List]
            .separate(inputFolders.toList.map{ p => prepareSubfolder(keepFile)(p, filenameFieldSep).map(p -> _) })
            .bimap(
                _.toNel.map(_.flatten), 
                folderContentPairs => 
                    val (unusables, prepped) = Alternative[List].separate(
                        folderContentPairs.map{ (subfolder, contents) => 
                            ensureNonemptyAndContinuous(contents).bimap(
                                UnparseablePathException(subfolder, _), 
                                _ -> contents
                                )
                        }
                    )
                    unusables.toNel.toLeft(prepped)
            ) match {
                case (Some(unusableSubfolderErrors), _) => unusableSubfolderErrors.asLeft
                case (None, preppedSubfolders) => preppedSubfolders.map{
                    case Nil => Nil
                    case (_, result) :: Nil => result
                    case subs@((n1, first) :: rest) => rest.foldLeft(n1 -> first.toVector){
                        case ((accCount, timePathPairs), (n, sub)) => 
                            val newCount = accCount `add` n
                            val newPairs = timePathPairs ++ sub.map(_.leftMap(t => Timepoint(t.get `add` accCount)))
                            newCount -> newPairs
                    }._2.toList
                }
            }
    }

    // Add a pair of nonnegative numbers, ensuring that the result stays as a nonnegative.
    extension (n: NonnegativeInt)
        infix def add(m: NonnegativeInt): NonnegativeInt = NonnegativeInt.either(n + m) match {
            case Left(msg) => throw new ArithmeticException(s"Uh-Oh! $n + $m = ${n + m}; $msg")
            case Right(result) => result
        }

    /** 
     * Ensure subfolder contents are nonempty, and that the timepoints are continuous starting from 0.
     * 
     * @param subfolderContents Collection of pairs of timepoint and path from which it was parsed, all from the same subfolder
     * @return Either an error message or the number of unique timepoints
     */
    def ensureNonemptyAndContinuous(subfolderContents: List[(Timepoint, os.Path)]): Either[String, NonnegativeInt] = 
        validateContinuityFromZero(subfolderContents.map(_._1))
            .map{ maxTime => maxTime.get `add` NonnegativeInt(1) }

    /** Select files to use, and map to {@code Right}-wrapped collection of pairs of time and path, or {@code Left}-wrapped errors. */
    def prepareSubfolder(keepFile: os.Path => Boolean)(folder: os.Path, filenameFieldSep: String): Either[NEL[UnparseablePathException], List[(Timepoint, os.Path)]] = {
        val (bads, goods) = 
            Alternative[List].separate(os.list(folder).toList
                .filter(f => os.isFile(f) && keepFile(f))
                .map{ p => Timepoint.parseValueIndexPairFromPath(p, filenameFieldSep).bimap(UnparseablePathException(p, _), _._1 -> p) }
            )
        bads.toNel.toLeft(goods)
    }

    /** Check that the sequence of timepoints is continuous and starts from 0. */
    def validateContinuityFromZero(ts: Seq[Timepoint]): Either[String, Timepoint] = {
        val raws = SortedSet(ts.map(_.get)*)
        Try { raws.max }.toEither.leftMap(_ => "Empty set of timepoints!").flatMap{ maxTime => 
            val missing = (0 to maxTime).map(NonnegativeInt.unsafe)
            missing.isEmpty.either(s"${missing.length} missing timepoints: $missing", Timepoint(maxTime))
        }
    }

    type Extension = String

    final case class UnparseablePathException(path: os.Path, message: String) 
        extends Exception(s"$path: $message")
    
    final case class UnusableSubfolderException(path: os.Path, message: String) 
        extends Exception(s"$path: $message")

    final case class UnusableTimepointUpdateException(path: os.Path, time: Timepoint, message: String) 
        extends Exception(s"($path, $time): $message")
end CombineImagingFolders