package at.ac.oeaw.imba.gerlich.looptrace
package syntax

trait BooleanRelatedSyntax:
    /** When an iterable is all booleans, simplify the all-true check ({@code ps.forall(identity) === ps.all}) */
    extension (ps: Iterable[Boolean])
        def all: Boolean = ps.forall(identity)
        def any: Boolean = ps.exists(identity)

