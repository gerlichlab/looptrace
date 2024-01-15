package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.syntax.all.*

enum Delimiter(val sep: String, val ext: String):
    case CommaSeparator extends Delimiter(",", "csv")
    case TabSeparator extends Delimiter("\t", "tsv")

    def canonicalExtension: String = ext
    def join(fields: Array[String]): String = fields mkString sep
    def filepath(folder: os.Path, baseName: String): os.Path = folder / s"${baseName}.${canonicalExtension}"
    def split(s: String): Array[String] = split(s, -1)
    def split(s: String, limit: Int): Array[String] = s.split(sep, limit)
end Delimiter

object Delimiter:
    given eqForDelimiter: Eq[Delimiter] = Eq.fromUniversalEquals[Delimiter]
    def fromPath(p: os.Path): Option[Delimiter] = fromExtension(p.ext)
    def fromPathUnsafe = (p: os.Path) => 
        fromPath(p).getOrElse{ throw new IllegalArgumentException(s"Cannot infer delimiter from file: $p") }
    def fromExtension(ext: String): Option[Delimiter] = Delimiter.values.filter(_.ext === ext).headOption
end Delimiter
