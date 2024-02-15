package at.ac.oeaw.imba.gerlich.looptrace

import cats.syntax.all.*

/** A round of imaging during an experiment */
sealed trait ImagingRound:
    def name: String
    def time: Timepoint
end ImagingRound

/**
 * An imaging round in which no hybridisation is done
 * 
 * @param name The name for the imaging round within an experiment
 * @param time The timepoint (0-based, inclusive) of this imaging round within an experiment
 */
final case class BlankImagingRound(name: String, time: Timepoint) extends ImagingRound

/** An imaging round in which FISH is done */
sealed trait FishImagingRound extends ImagingRound:
    /** Name for the probe used for hybridisation */
    def probe: ProbeName
    /** Number (1-based, inclusive) of repeat of a particular probe that this round represents within the experiment */
    def repeat: Option[PositiveInt]
end FishImagingRound

/**
  * Round of imaging in which a specific locus was targeted during experiment
  *
  * @param name Name for the imaging round
  * @param time The timepoint (0-based, inclusive) of this imaging round within an experiment
  * @param probe Name for the probe used for hybridisation
  * @param repeat  Number (1-based, inclusive) of repeat of a particular probe that this round represents within the experiment
  */
final case class LocusImagingRound(name: String, time: Timepoint, probe: ProbeName, repeat: Option[PositiveInt]) extends FishImagingRound:
    final def isRepeat = repeat.nonEmpty

/** Helpers and alternate constructors for working with imaging rounds of specific genomic loci */
object LocusImagingRound:
    def apply(time: Timepoint, probe: ProbeName): LocusImagingRound = apply(None, time, probe, None)
    def apply(time: Timepoint, probe: ProbeName, repeat: PositiveInt): LocusImagingRound = apply(None, time, probe, repeat.some)
    def apply(maybeName: Option[String], time: Timepoint, probe: ProbeName, maybeRepeat: Option[PositiveInt]): LocusImagingRound = 
        val name = maybeName.getOrElse(probe.get ++ maybeRepeat.fold("")(rep => s"_repeat${rep.show}"))
        new LocusImagingRound(name, time, probe, maybeRepeat)
end LocusImagingRound

/**
  * Round of imaging in which an entire region (multiple individual loci) was targeted during experiment
  *
  * @param name Name for the imaging round
  * @param time The timepoint (0-based, inclusive) of this imaging round within an experiment
  * @param probe Name for the probe used for hybridisation
  */
final case class RegionalImagingRound(name: String, time: Timepoint, probe: ProbeName) extends FishImagingRound:
    override final def repeat: Option[PositiveInt] = None
end RegionalImagingRound

/** Helpers and alternate constructors for working with imaging rounds of genomic regions */
object RegionalImagingRound:
    def apply(time: Timepoint, probe: ProbeName): RegionalImagingRound = apply(None, time, probe)
    def apply(maybeName: Option[String], time: Timepoint, probe: ProbeName): RegionalImagingRound = 
        new RegionalImagingRound(maybeName.getOrElse(probe.get), time, probe)
end RegionalImagingRound
