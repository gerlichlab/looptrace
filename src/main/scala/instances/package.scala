package at.ac.oeaw.imba.gerlich.looptrace

/** Typeclass instances */
package object instances:
  /** An aggregator of typeclass instances for custom data types, to facilitate
    * import ... .all.given
    */
  object all extends AllInstances

  /** A bundle of typeclass instances for many of this package's custom data
    * types
    */
  private trait AllInstances
      extends LocusIdInstances,
        ProbeNameInstances,
        RegionIdInstances,
        RoiIdInstances,
        TraceIdInstances
