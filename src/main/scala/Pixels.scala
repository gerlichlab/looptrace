package at.ac.oeaw.imba.gerlich.looptrace

import squants.MetricSystem
import squants.space.{ Length, LengthUnit }
import at.ac.oeaw.imba.gerlich.gerlib.numeric.PositiveReal

/** A fundamental unit of length in imaging, the pixel */
object Pixel:
    /** Define a unit of length in pixels by specifying number of nanometers per pixel. */
    def defineByNanometers(nmPerPx: PositiveReal): LengthUnit = new:
        val conversionFactor: Double = nmPerPx * MetricSystem.Nano
        val symbol: String = "px"
end Pixel