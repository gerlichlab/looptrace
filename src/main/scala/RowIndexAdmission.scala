package at.ac.oewa.imba.gerlich.looptrace

import cats.*
import at.ac.oeaw.imba.gerlich.gerlib.numeric.NonnegativeInt

/** Evidence that an {@code Out}-wrapped nonnegative integer can be obtained from an {@code In} */
trait RowIndexAdmission[In, Out[_]]:
    def getRowIndex: In => Out[NonnegativeInt]
end RowIndexAdmission

/** Tools for working with getters of row index */
object RowIndexAdmission:
    given contravariantForRowIndexAdmission[Out[_]]: Contravariant[[In] =>> RowIndexAdmission[In, Out]] with
        override def contramap[A, B](fa: RowIndexAdmission[A, Out])(f: B => A): RowIndexAdmission[B, Out] = new:
            override def getRowIndex: B => Out[NonnegativeInt] = f `andThen` fa.getRowIndex

    def intoIdentity[I](f: I => NonnegativeInt): RowIndexAdmission[I, Id] = new:
        override def getRowIndex: I => Id[NonnegativeInt] = f

    def idLeftForTuple2[A]: RowIndexAdmission[(NonnegativeInt, A), Id] = intoIdentity((_: (NonnegativeInt, A))._1)

    def idRightForTuple2[A]: RowIndexAdmission[(A, NonnegativeInt), Id] = intoIdentity((_: (A, NonnegativeInt))._2)
end RowIndexAdmission
