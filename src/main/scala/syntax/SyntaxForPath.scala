package at.ac.oeaw.imba.gerlich.looptrace
package syntax

trait SyntaxForPath:
  /** Add a {@code .parent} accessor on a path. */
  extension (p: os.Path) def parent: os.Path = p / os.up
