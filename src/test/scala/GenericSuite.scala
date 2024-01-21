package at.ac.oeaw.imba.gerlich.looptrace

import org.scalacheck.{ Arbitrary, Gen }
import org.scalacheck.Arbitrary.arbitrary
import com.github.tototoshi.csv.*

/** Tools very generally useful in automated testing */
trait GenericSuite:

    /***********************/
    /* Arbitrary instances */
    /***********************/
    given nonnegativeIntArbitray: Arbitrary[NonnegativeInt] = Arbitrary(genNonnegativeInt)
    given positiveIntArbitrary: Arbitrary[PositiveInt] = Arbitrary(genPositiveInt)
    given nonnegativeRealArbitrary: Arbitrary[NonnegativeReal] = Arbitrary(genNonnegativeReal)

    /********************/
    /* Generators       */
    /********************/
    def genJsonInt: Gen[ujson.Num] = arbitrary[Int].map(ujson.Num.apply compose (_.toDouble))
    def genNonnegativeInt: Gen[NonnegativeInt] = Gen.choose(0, Int.MaxValue).map(NonnegativeInt.unsafe)
    def genNonnegativeReal: Gen[NonnegativeReal] = Gen.choose(0.0, Double.MaxValue).map(NonnegativeReal.unsafe)
    def genPositiveInt: Gen[PositiveInt] = Gen.posNum[Int].map(PositiveInt.unsafe)

    /********************/
    /* Other defintions */
    /********************/
    
    /** Create the given file with empty string as content. */
    def touchFile(fp: os.Path, createFolders: Boolean = false): Unit = 
        os.write(fp, "", createFolders = createFolders)

    /** Execute some test code that uses a {@code os.Path} folder. */
    def withTempDirectory(testCode: os.Path => Any): Any = {
        val tempRoot = os.temp.dir()
        try { testCode(tempRoot) } finally { os.remove.all(tempRoot) }
    }

    def withTempFile(delimiter: Delimiter)(testCode: os.Path => Any): Any = 
        withTempFile(initData = null, delimiter = delimiter)(testCode)

    def withTempFile(initData: os.Source, delimiter: Delimiter)(testCode: os.Path => Any): Any = 
        withTempFile(initData = initData, suffix = "." ++ delimiter.ext)

    /** Execute some test code that uses a {@code os.Path} file. */
    def withTempFile(initData: os.Source = null, suffix: String = "")(testCode: os.Path => Any): Any = {
        val tempfile = os.temp(contents = initData, suffix = suffix)
        try { testCode(tempfile) } finally { os.remove(tempfile) }
    }

    /** Execute some test code that uses a JSON file {@code os.Path} file. */
    def withTempJsonFile(initData: os.Source, suffix: String = ".json") = withTempFile(initData, suffix)(_: os.Path => Any)
end GenericSuite
