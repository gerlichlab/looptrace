package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.{ ValidatedNel }
import cats.syntax.all.*
import mouse.boolean.*
import upickle.default.*

import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.UJsonHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.*

/**
 * Tools for analysing chromatin fiber tracing (typically *_traces*.csv)
 * 
 * Originally developed for Neos, to count the number of locus-specific spots 
 * kept on a per-(ref_frame, frame) basis. That is, how many locus-specific 
 * spots are available (after QC filtering) per specific target (using a 
 * two-stage multiplex).
 * 
 * @author Vince Reuter
 */
object TracingOutputAnalysis:
    
    /** Pairing or regional and locus-specific FISH spot, a common key by which to group and/or filter records */
    type SpotTimePair = (RegionalSpotTimepoint, LocusSpotTimepoint)

    /** Helpers for working with pairs of regional and locus-specific timepoint */
    object SpotTimePair:
        /** Key in JSON for a value representing regional spot imaging timepoint */
        private[TracingOutputAnalysis] val regionalKey = "regional"
        /** Key in JSON for a value representing locus-specific spot imaging timepoint */
        private[TracingOutputAnalysis] val localKey = "local"

        /** Order by the numeric values, component-wise. */
        given orderForSpotTimePair: Order[SpotTimePair] = Order.by{ 
            case (RegionalSpotTimepoint(r), LocusSpotTimepoint(l)) => r -> l
        }

        /** JSON codec for pair of regional spot timepoint and locus spot timepoint */
        given rwForSpotTimePair: ReadWriter[SpotTimePair] = readwriter[ujson.Value].bimap(
            { case (RegionalSpotTimepoint(FrameIndex(r)), LocusSpotTimepoint(FrameIndex(l))) => 
                ujson.Obj(regionalKey -> ujson.Num(r), localKey -> ujson.Num(l))
            },
            json => {
                val regional = RegionalSpotTimepoint(FrameIndex.unsafe(json(regionalKey).int))
                val local = LocusSpotTimepoint(FrameIndex.unsafe(json(localKey).int))
                regional -> local
            }
        )
    end SpotTimePair

    /** Write a collection of record counts by (region, locus) to given output file. */
    private[looptrace] def writeCountsToCsv(
        counts: Iterable[(SpotTimePair, Int)], 
        outfile: os.Path, 
        )(using ev: Order[SpotTimePair]): Unit = {
        val sep = Delimiter.CommaSeparator
        if (outfile.ext =!= sep.ext) {
            throw new IllegalArgumentException(
                s"Unexpected extension ('${outfile.ext}', not '${sep.ext}') for output file: $outfile"
            )
        }
        val header = Array(SpotTimePair.regionalKey, SpotTimePair.localKey, "N")
        val fieldRecs = counts.toList
            .sortBy(_._1)(ev.toOrdering)
            .map{ case ((r, l), n) => Array(r.get.get, l.get.get, n).map(_.show) }
        val lines = (header :: fieldRecs).map(sep.join(_) ++ "\n")
        os.write(outfile, lines)
    }

    /** Wrap a {@code FrameIndex} as signifying the timepoint of a regional spot. */
    final case class RegionalSpotTimepoint(get: FrameIndex)
    
    /** Tools for building and working with {@code RegionalSpotTimepoint} values */
    object RegionalSpotTimepoint:
        given orderForRegionalSpotTimepoint: Order[RegionalSpotTimepoint] = Order.by(_.get)
        private[TracingOutputAnalysis] def getFromRow(row: CsvRow): ValidatedNel[String, RegionalSpotTimepoint] = 
            getFromRowThroughFrameIndex(row, "ref_frame")(RegionalSpotTimepoint.apply)
        private[TracingOutputAnalysis] def getFromRowUnsafe(row: CsvRow): RegionalSpotTimepoint = 
            getFromRow(row).fold(
                errs => throw new Exception(s"Problem(s) parsing regional spot timepoint from row ($row): $errs"), 
                identity
                )
    end RegionalSpotTimepoint

    /** Wrap a {@code FrameIndex} as signifying the timepoint of a locus-specific spot. */
    final case class LocusSpotTimepoint(get: FrameIndex)
    
    /** Tools for building and working with {@code LocusSpotTimepoint} values */
    object LocusSpotTimepoint:
        given orderForLocusSpotTimepoint: Order[LocusSpotTimepoint] = Order.by(_.get)
        private[TracingOutputAnalysis] def getFromRow(row: CsvRow): ValidatedNel[String, LocusSpotTimepoint] = 
            getFromRowThroughFrameIndex(row, "frame")(LocusSpotTimepoint.apply)
        private[TracingOutputAnalysis] def getFromRowUnsafe(row: CsvRow): LocusSpotTimepoint = 
            getFromRow(row).fold(
                errs => throw new Exception(s"Problem(s) parsing locus spot timepoint from row ($row): $errs"), 
                identity
                )
    end LocusSpotTimepoint

    /**
      * A typeclass representing how to build a (positive) selector for elements from an arbitary pool
      * 
      * @tparam Pool Type of value which will be queried for membership
      * @tparam Elem Type of value which will be queried for selection
      */
    trait ElementSelectorBuilder[Pool, Elem]:
        /** How to build a (positive) selection function from a given "pool" of values */
        def buildSelector: Pool => Elem => Boolean
    end ElementSelectorBuilder

    /** Tools for working with construction of element selectors */
    object ElementSelectorBuilder:
        /** A set's builder defines selection/inclusion as membership in the set if nonempty, always-true no-op for empty set. */
        given builderForEmptySet[A]: ElementSelectorBuilder[Set[A], A] with
            def buildSelector = (pool: Set[A]) => if pool.isEmpty then Function.const(true) else pool.contains
    end ElementSelectorBuilder

    /** Alias for positive selection/inclusion function for pairs of regional and local spot timepoint */
    type RegLocFilter = SpotTimePair => Boolean
    
    /** Helpers for building and working with filters of pairs of regional and locus-specific image timepoints. */
    object RegLocFilter:
        import ElementSelectorBuilder.given
        
        /** General result of trying to build a filter instance from read data, either an error message or an instance */
        private type ParseResult = Either[String, RegLocFilter]
        
        /** Read from JSON by parsing to list of pairs, then checking for uniqueness. */
        given rwForPairFilter(using rwRegLocPair: ReadWriter[SpotTimePair]): ReadWriter[RegLocFilter] = 
            readwriter[List[SpotTimePair]].bimap(
                _ => throw new NotImplementedError("Cannot serialise a spot time pair filter!"),
                fromList(_).fold(msg => throw new NonUniquenessException(msg), identity)
            )
        
        /** Check for uniqueness before building the filter instance; use {@code Order[SpotTimePait]} to check equality. */
        private def fromList(pairs: List[SpotTimePair])(using ev: ElementSelectorBuilder[Set[SpotTimePair], SpotTimePair]): ParseResult = {
            val uniq = pairs.toSet
            (uniq.size === pairs.length).either(s"Collection to parse contains at least 1 repeat! $pairs", ev.buildSelector(uniq).apply)
        }
        
        /**
         * Read a list of (regional, local) pairs to build a filter that will include items with any pair parsed here.
         * 
         * @param f The file (CSV) to parse
         * @throws NonUniquenessException if the file parses but gives records with at least 1 repeat
         * @throws java.lang.Exception if the file's first line doesn't parse to expected header
         * @throws Throwable
         */
        def fromCsvFileUnsafe(f: os.Path): RegLocFilter = {
            val expectedHeader =  List("ref_frame", "frame")
            val (head, rows) = safeReadAllWithOrderedHeaders(f).fold(throw _, identity)
            if (head =!= expectedHeader)
                throw new Exception(s"Unexpected header ($head) from regional/local pairs filter file ($f)! Expected: $expectedHeader")
            fromList(rows.map(r => RegionalSpotTimepoint.getFromRowUnsafe(r) -> LocusSpotTimepoint.getFromRowUnsafe(r))) match {
                case Left(msg) => throw new NonUniquenessException(msg)
                case Right(filt) => filt
            }
        }

        /** Read a list of (regional, local) pairs to build a filter that will include items with any pair parsed here. */
        def fromJsonFileUnsafe(f: os.Path)(using ev: Reader[RegLocFilter]) = readJsonFile[RegLocFilter](f)
        
        /** Error type for when values that should be unique aren't */
        final case class NonUniquenessException(message: String) extends Exception(message)
    end RegLocFilter

    /**
      * Write the record count for each (regional, local) pair of interest.
      *
      * @param pairsOfInterestFile Path to the file listing pairs of regional and local spot images
      *     of interst
      * @param infile Path to the file in which to count records
      * @param outfile Path to file to which to write counts
      */
    def writeRegionalLocalPairCountsFiltered(pairsOfInterestFile: os.Path)(
        infile: os.Path, 
        outfile: os.Path, 
        ): Unit = {
        given eqForPath: Eq[os.Path] = Eq.by(_.toString)
        require(
            pairsOfInterestFile =!= infile && infile =!= outfile && pairsOfInterestFile =!= outfile, 
            s"Non-unique filepath arguments: ($pairsOfInterestFile, $infile, $outfile)"
            )
        val counts = countByRegionLocusPairUnsafe(pairsOfInterestFile = pairsOfInterestFile, dataFile = infile)
        writeCountsToCsv(counts.toList, outfile = outfile)
    }

    /**
      * Write the record count for each (regional, local) pair of interest.
      *
      * @param pairsOfInterestFile Path to the file (JSON or CSV) listing pairs 
      *     of regional and local spot images of interst
      * @param infile Path to the file in which to count records
      * @param outfile Path to file to which to write counts
      */
    def writeRegionalLocalPairCounts(
        infile: os.Path, 
        outfile: os.Path, 
        ): Unit = {
        given orderForPath: Order[os.Path] = Order.by(_.toString)
        require(infile =!= outfile, s"Input and output files match: ($infile, $outfile)")
        val counts = countByRegionLocusPairUnsafe(infile)
        writeCountsToCsv(counts.toList, outfile = outfile)
    }

    /**
      * Count records by pair of regional and locus-specific spot image timepoint.
      *
      * @param pairsOfInterestFile Path to the file (JSON or CSV) listing pairs 
      *     of regional and local spot images of interst
      * @param dataFile Path to file in which to count grouped records
      * @return Mapping from pair of regional and locus-specific imaging timepoints 
      *     to count of records from given {@code dataFile} with that pair of times
      */
    def countByRegionLocusPairUnsafe(pairsOfInterestFile: os.Path, dataFile: os.Path): Map[SpotTimePair, Int] = {
        import RegLocFilter.*
        import RegLocFilter.given
        import SpotTimePair.given
        println(s"Reading pairs of interest: $pairsOfInterestFile")
        val useRegLoc = pairsOfInterestFile.ext match {
            case "csv" => RegLocFilter.fromCsvFileUnsafe(pairsOfInterestFile)
            case "json" => RegLocFilter.fromJsonFileUnsafe(pairsOfInterestFile)
            case ext => throw new IllegalArgumentException(s"Cannot parse pairs of interest file with extension '$ext': $pairsOfInterestFile")
        }
        println(s"Reading traces file: $dataFile")
        val counts: Map[SpotTimePair, Int] = countByRegionLocusPairUnsafe(dataFile)
        counts.view.filterKeys(useRegLoc).toMap
    }

    /**
      * Count records by pair of regional and locus-specific spot image timepoint.
      *
      * @param dataFile Path to file in which to count grouped records
      * @return Mapping from pair of regional and locus-specific imaging timepoints 
      *     to count of records from given {@code dataFile} with that pair of times
      */
    def countByRegionLocusPairUnsafe(f: os.Path): Map[SpotTimePair, Int] = 
        safeReadAllWithHeaders(f).fold(throw _, countByRegionLocusPairUnsafe)

    private def countByKey[R, K : Order](records: Iterable[R])(key: R => K): Map[K, Int] = 
        records.groupBy(key).view.mapValues(_.size).toMap
    
    private def countByRegionLocusPairUnsafe(rows: Iterable[CsvRow]): Map[SpotTimePair, Int] = countByKey(rows){ row => 
        val regional = RegionalSpotTimepoint.getFromRowUnsafe(row)
        val local = LocusSpotTimepoint.getFromRowUnsafe(row)
        regional -> local
    }

    private def parseThroughFrameIndex[A](build: FrameIndex => A): String => Either[String, A] = 
        safeParseInt >>> NonnegativeInt.either >> FrameIndex.apply >> build

    private def getFromRowThroughFrameIndex[A](row: CsvRow, key: String)(build: FrameIndex => A): ValidatedNel[String, A] = 
        safeGetFromRow(key, parseThroughFrameIndex(build))(row)

end TracingOutputAnalysis
