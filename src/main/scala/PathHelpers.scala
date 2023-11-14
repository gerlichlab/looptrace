package at.ac.oeaw.imba.gerlich.looptrace

/** Utility functions for working with paths */
object PathHelpers:

    /** Collect all the paths which share the given root. */
    def listPath(root: os.Path): Vector[os.Path] = {
        var paths: Vector[os.Path] = Vector()
        os.proc("find", root).call(cwd = os.pwd, stdout = os.ProcessOutput.Readlines(line => paths = paths :+ os.Path(line)))
        paths
    }
