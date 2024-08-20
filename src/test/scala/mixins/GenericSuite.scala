package at.ac.oeaw.imba.gerlich.looptrace

import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary
import com.github.tototoshi.csv.*

import at.ac.oeaw.imba.gerlich.gerlib.testing.instances.NumericInstances

/** Tools very generally useful in automated testing */
trait GenericSuite extends NumericInstances:    
    /** Create the given file with empty string as content. */
    def touchFile(fp: os.Path, createFolders: Boolean = false): Unit = 
        os.write(fp, "", createFolders = createFolders)

    /** Execute some test code that uses a {@code os.Path} folder. */
    def withTempDirectory(testCode: os.Path => Any): Any = {
        val tempRoot = os.temp.dir()
        try { testCode(tempRoot) } finally { os.remove.all(tempRoot) }
    }

    /** NB: delimiter is only used to determine the filename extension. */
    def withTempFile(delimiter: Delimiter)(testCode: os.Path => Any): Any = 
        withTempFile(initData = null, delimiter = delimiter)(testCode)

    /** NB: delimiter is only used to determine the filename extension. */
    def withTempFile(initData: os.Source, delimiter: Delimiter)(testCode: os.Path => Any): Any = 
        withTempFile(initData = initData, suffix = "." ++ delimiter.ext)(testCode)

    /** Execute some test code that uses a {@code os.Path} file. */
    def withTempFile(initData: os.Source = null, suffix: String = "")(testCode: os.Path => Any): Any = {
        val tempfile = os.temp(contents = initData, suffix = suffix)
        try { testCode(tempfile) } finally { os.remove(tempfile) }
    }

    /** Execute some test code that uses a JSON file {@code os.Path} file. */
    def withTempJsonFile(initData: os.Source, suffix: String = ".json") = 
        withTempFile(initData, suffix)(_: os.Path => Any)
end GenericSuite
