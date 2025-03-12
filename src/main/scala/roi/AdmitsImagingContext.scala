package at.ac.oeaw.imba.gerlich.looptrace
package roi

import cats.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext

trait AdmitsImagingContext[T]:
    def getImagingContext: T => ImagingContext
end AdmitsImagingContext

/** Helpers for working with access to a [[at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingContext]] value */
object AdmitsImagingContext:
    def instance[T](f: T => ImagingContext): AdmitsImagingContext[T] = new:
        override def getImagingContext: T => ImagingContext = f

    given Contravariant[AdmitsImagingContext]:
        override def contramap[A, B](fa: AdmitsImagingContext[A])(f: B => A): AdmitsImagingContext[B] = new:
            override def getImagingContext: B => ImagingContext = f `andThen` fa.getImagingContext

    extension [T](t: T)
        def imagingContext(using ev: AdmitsImagingContext[T]): ImagingContext = ev.getImagingContext(t)
end AdmitsImagingContext
