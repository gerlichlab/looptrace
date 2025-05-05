package at.ac.oewa.imba.gerlich.looptrace

import cats.*
import io.github.iltotore.iron.:|
import io.github.iltotore.iron.constraint.any.Not
import io.github.iltotore.iron.constraint.numeric.Negative

/** Evidence that an {@code Out}-wrapped nonnegative integer can be obtained
  * from an {@code In}
  */
trait RowIndexAdmission[In, Out[_]]:
  def getRowIndex: In => Out[Int :| Not[Negative]]
end RowIndexAdmission

/** Tools for working with getters of row index */
object RowIndexAdmission:
  given [Out[_]] => Contravariant[[In] =>> RowIndexAdmission[In, Out]]:
    override def contramap[A, B](fa: RowIndexAdmission[A, Out])(
        f: B => A
    ): RowIndexAdmission[B, Out] = new:
      override def getRowIndex: B => Out[Int :| Not[Negative]] =
        f `andThen` fa.getRowIndex

  def intoIdentity[I](f: I => Int :| Not[Negative]): RowIndexAdmission[I, Id] =
    new:
      override def getRowIndex: I => Id[Int :| Not[Negative]] = f

  def idLeftForTuple2[A]: RowIndexAdmission[(Int :| Not[Negative], A), Id] =
    intoIdentity((_: (Int :| Not[Negative], A))._1)

  def idRightForTuple2[A]: RowIndexAdmission[(A, Int :| Not[Negative]), Id] =
    intoIdentity((_: (A, Int :| Not[Negative]))._2)
end RowIndexAdmission
