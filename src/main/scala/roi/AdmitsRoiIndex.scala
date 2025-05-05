package at.ac.oeaw.imba.gerlich.looptrace
package roi

trait AdmitsRoiIndex[T]:
  def getRoiIndex: T => RoiIndex
end AdmitsRoiIndex

object AdmitsRoiIndex:
  extension [T](t: T)
    def roiIndex(using ev: AdmitsRoiIndex[T]): RoiIndex = ev.getRoiIndex(t)

  def instance[T](f: T => RoiIndex): AdmitsRoiIndex[T] = new:
    override def getRoiIndex: T => RoiIndex = f
end AdmitsRoiIndex
