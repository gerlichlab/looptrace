package at.ac.oeaw.imba.gerlich.looptrace
package syntax

trait SyntaxForFunction:
  /** Add a continuation-like syntax for flatmapping over a function that can
    * fail with another that can fail.
    */
  extension [A, L, B, R](f: A => Either[L, B])
    infix def >>>[L1 >: L](g: B => Either[L1, R]): A => Either[L1, R] =
      f(_: A).flatMap(g)

  /** Add a continuation-like syntax for flatmapping over a function that can
    * fail with another that canNOT fail.
    */
  extension [A, L, B](f: A => Either[L, B])
    infix def >>[C](g: B => C): A => Either[L, C] = f(_: A).map(g)

  extension [I, O](f: I => Unit)
    def returning(o: O): I => O = f `andThen` Function.const(o)
