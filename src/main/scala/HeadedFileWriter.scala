package at.ac.oeaw.imba.gerlich.looptrace

import java.io.{ BufferedWriter, File, FileWriter }
import cats.*
import cats.syntax.all.*
import mouse.boolean.*

/**
 * A type which writes values of a particular type to a delimited text file with a header
 * 
 * This makes it easier to keep correct the pairing between delimiter and file extension, 
 * as well as the match between number of fields in a header and number of fields in 
 * each line of text to be written to a file, to avoid nasty data integrity errors.
 * 
 * @tparam R The type of value representing each record to write to text file
 * @author Vince Reuter
 */
trait HeadedFileWriter[R]:
    def header: Array[String]
    
    def toTextFields(r: R): Array[String]
    
    def delimiter: Delimiter
    
    def toTextFieldsChecked(r: R): Either[String, Array[String]] = {
        val fields = toTextFields(r)
        (fields.length === header.length).either(s"${fields.length} field(s) in record and ${header.length} in header!", fields)
    }
    def toTextFieldsCheckedUnsafe(r: R): Array[String] = toTextFieldsChecked(r).fold(msg => throw new Exception(msg), identity)

    /**
      * Write the given collection of records to the given target.
      *
      * @tparam T The type of target value
      * @param records The records to write to the given target
      * @param target The "sink", or where to write the given records
      * @param ev Evidence of a {@code HeadedFileWriter.Target} instance for type {@code T} available in scope
      * @return The given target
      */
    def writeRecordsToFile[T](records: Iterable[R])(target: T)(implicit ev: HeadedFileWriter.Target[T]): T = {
        val outfile = ev.getFile(target)
        println(s"Writing records to file: $outfile")
        val fieldsToLine = ev.getLineMaker(target)
        os.write(outfile, records.map{ r => fieldsToLine(toTextFieldsCheckedUnsafe(r)) ++ "\n" })
        println("Done!")
        target
    }
end HeadedFileWriter

/**
 * Tools for working with writing delimited text files with a header
 * 
 * @author Vince Reuter
 */
object HeadedFileWriter:
    /** A target / output file for some file writing process, bundling a filepath, and line formation */
    trait Target[A] {
        /** How to convert a record (sequence of text fields) into a line of text to write */
        def getLineMaker: A => (Array[String] => String)
        /** File to which to write */
        def getFile: A => os.Path
    }

    /** Syntax enrichment for any value of a type for which there's a {@code Target} instance available */
    extension [A](a: A)(using ev: Target[A])
        def fieldsToLine: Array[String] => String = ev.getLineMaker(a)
        def value: os.Path = ev.getFile(a)
        final def isFile: Boolean = os.isFile(value)

    final case class DelimitedTextTarget(folder: os.Path, nameBase: String, delimiter: Delimiter):
        def fieldsToLine: Array[String] => String = delimiter.join
        def filepath: os.Path = delimiter.filepath(folder, nameBase)
    end DelimitedTextTarget

    /** Helpers for working with delimited text file targets */
    object DelimitedTextTarget:
        /** Provide an instance of the target-like typeclass for this more specific type. */
        given TargetForDelimitedTextTarget: Target[DelimitedTextTarget] with
            override def getLineMaker: DelimitedTextTarget => (Array[String] => String) = _.fieldsToLine
            override def getFile: DelimitedTextTarget => os.Path = _.filepath
        given eqForDelimitedTextTarget(using Eq[os.Path]): Eq[DelimitedTextTarget] = Eq.by{
            target => (target.folder, target.nameBase, target.delimiter)
        }
    end DelimitedTextTarget
end HeadedFileWriter
