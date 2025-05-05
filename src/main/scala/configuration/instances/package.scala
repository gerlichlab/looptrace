package at.ac.oeaw.imba.gerlich.looptrace.configuration

/** Configuration-related typeclass instances */
package object instances:
  object all extends AllConfigurationInstances

  trait AllConfigurationInstances
      extends PureConfigLooptraceInstances,
        PureConfigSquantsInstances
end instances
