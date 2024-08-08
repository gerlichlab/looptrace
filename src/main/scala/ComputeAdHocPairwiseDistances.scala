package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.syntax.all.*
import mouse.boolean.*
import scopt.OParser
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.looptrace.CsvHelpers.*
import at.ac.oeaw.imba.gerlich.looptrace.cli.ScoptCliReaders
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.space.*

/**
 * Quick ad-hoc program for computing pairwise Euclidean distances
 * 
 * Input data should be a CSV, with first line being header and each other 
 * line being a data row. The header should have an ID column, and a 
 * column for Z, for Y, and for X. These have defaults but are 
 * configurable via the CLI.
 * 
 * @author Vince Reuter
 */
object ComputeAdHocPairwiseDistances extends ScoptCliReaders with StrictLogging:
    private val ProgramName = "ComputeAdHocPairwiseDistances"

    final case class CliConfig(
        infile: os.Path = null, // unconditionally required
        outfile: os.Path = null, // unconditionally required
        idKey: String = "Frames",
        zKey: String = "Z", 
        yKey: String = "Y", 
        xKey: String = "X",
    )
    val cliParseBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import cliParseBuilder.*
        
        val parser = OParser.sequence(
            programName(ProgramName),
            head(ProgramName, BuildInfo.version),
            opt[os.Path]('I', "infile")
                .required()
                .action((f, c) => c.copy(infile = f))
                .validate(f => os.isFile(f).either(s"Input isn't a file: $f", ()))
                .text("Path to input file"),
            opt[os.Path]('O', "outfile")
                .required()
                .action((f, c) => c.copy(outfile = f))
                .text("Path to output file"),
            opt[String]("idKey")
                .action((k, c) => c.copy(idKey = k))
                .text("Key for the ID column of the tabular data"),
            opt[String]("idKey")
                .action((k, c) => c.copy(idKey = k))
                .text("Key for the ID column of the tabular data"),
            opt[String]("zKey")
                .action((k, c) => c.copy(zKey = k))
                .text("Key for the Z column of the tabular data"),
            opt[String]("yKey")
                .action((k, c) => c.copy(yKey = k))
                .text("Key for the Y column of the tabular data"),
            opt[String]("xKey")
                .action((k, c) => c.copy(xKey = k))
                .text("Key for the X column of the tabular data"),
        )
        
        OParser.parse(parser, args, CliConfig()) match {
            case None => throw new Exception("Illegal use, check --help")
            case Some(opts) => 
                logger.info(s"Reading input file: ${opts.infile}")
                val (_, rows) = safeReadAllWithOrderedHeaders(opts.infile).fold(throw _, identity)
                val parseId = (r: Map[String, String]) => r(opts.idKey)
                val parsePoint = (r: Map[String, String]) => 
                    val z = ZCoordinate(r(opts.zKey).toDouble)
                    val y = YCoordinate(r(opts.yKey).toDouble)
                    val x = XCoordinate(r(opts.xKey).toDouble)
                    Point3D(x, y, z)
                val records = rows.map(r => parseId(r) -> parsePoint(r))
                val outs = records.combinations(2).map{
                    case (f1, p1) :: (f2, p2) :: Nil => 
                        val d = EuclideanDistance.between(p1, p2)
                        (f1, f2, d)
                    case combo => throw new Exception(s"Uh-OH, Alleged pairwise combo has ${combo.size} (not 2) elements!")
                }
                given showForEucl: Show[EuclideanDistance] = Show.show(_.get.toString)
                val outlines = outs.map((f1, f2, d) => List(f1.show, f2.show, d.show).mkString(","))
                logger.info(f"Writing output file: ${opts.outfile}")
                os.write.over(opts.outfile, outlines.map(_ ++ "\n"))
                logger.info("Done!")
        }
    }
end ComputeAdHocPairwiseDistances
