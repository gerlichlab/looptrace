package at.ac.oeaw.imba.gerlich.looptrace

/**
 * Math tools for testing
 *
 * @author Vince Reuter
 */
trait MathSuite:
    // nCk, i.e. number of ways to choose k indistinguishable objects from n
    extension (n: Int)
        infix protected def choose(k: Int): Int = {
            require(n >= 0 && n <= 10, s"n not in [0, 10] for nCk: $n")
            require(k <= n, s"Cannot choose more items than available: $k > $n")
            factorial(n) / (factorial(k) * factorial(n - k))
        }

    protected def factorial(n: Int): Int = {
        require(n >= 0 && n <= 10, s"Factorial arg not in [0, 10]: $n")
        (1 to n).product
    }
end MathSuite