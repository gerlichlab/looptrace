package at.ac.oeaw.imba.gerlich.looptrace

import java.io.File
import java.nio.file.FileAlreadyExistsException
import scopt.Read

/** How a program should handle trying to write output when the target already exists. */
enum ExtantOutputHandler:
    case Overwrite, Skip, Fail

    /** Print a message about why processing can proceed, or provide a {@code Left}-wrapped message or exception */
    def prepareToWrite(f: os.Path): Either[Unit | FileAlreadyExistsException, os.Path] = {
        if !os.isFile(f) then Right(f)
        else this match {
            case ExtantOutputHandler.Skip => Left(())
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
