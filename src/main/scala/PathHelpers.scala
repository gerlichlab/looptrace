package at.ac.oeaw.imba.gerlich.looptrace

/** Utility functions for working with paths */
object PathHelpers:

    /** Count the number of lines in the given file. */
    def countLines(f: os.Path): Int = os.read.lines(f).length

    /** Collect all the paths which share the given root. */
    def listPath(root: os.Path): Vector[os.Path] = {
        var paths: Vector[os.Path] = Vector()
        os.proc("find", root).call(cwd = os.pwd, stdout = os.ProcessOutput.Readlines(line => paths = paths :+ os.Path(line)))
        paths
    }

end PathHelpers
