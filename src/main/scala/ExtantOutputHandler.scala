package at.ac.oeaw.imba.gerlich.looptrace

import java.io.File
import java.nio.file.FileAlreadyExistsException
import cats.syntax.all.*
import scopt.Read

import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** How a program should handle trying to write output when the target already exists. */
enum ExtantOutputHandler:
    case Overwrite, Skip, Fail

    /** Get a total function taking an {@code os.Source} and producing a result saying if the target will be written. */
    def getSimpleWriter(target: os.Path): Either[FileAlreadyExistsException, os.Source => Boolean] = this match {
        case ExtantOutputHandler.Skip if os.isFile(target) => ((_: os.Source) => false).asRight
        case ExtantOutputHandler.Fail if os.isFile(target) => FileAlreadyExistsException(target.toString).asLeft
        case _ => (os.write.over(target, (_: os.Source))).returning(true).asRight
    }

    /** Print a message about why processing can proceed, or provide a {@code Left}-wrapped message or exception */
    def prepareToWrite(f: os.Path): Either[String | FileAlreadyExistsException, os.Path] = {
        if !os.isFile(f) then Right(f)
        else this match {
            case ExtantOutputHandler.Skip => Left(s"File to write already exists: $f")
            case ExtantOutputHandler.Fail => Left(new FileAlreadyExistsException(f.toString))
            case ExtantOutputHandler.Overwrite => Right(f)
        }
    }
end ExtantOutputHandler

/** Helpers for working with specification of how to handle extant output */
object ExtantOutputHandler:
    given ReadForExtantOutputHandler: scopt.Read[ExtantOutputHandler] =
        scopt.Read.reads((s: String) => parse(s).getOrElse { throw new Exception(s"Illegal extant output handler: $s") })
    def parse: String => Option[ExtantOutputHandler] = s => lookup.get(s.toLowerCase)
    def render(handler: ExtantOutputHandler): String = handler.toString.toLowerCase
    private def lookup = ExtantOutputHandler.values.map(x => render(x) -> x).toMap
end ExtantOutputHandler
