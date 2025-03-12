package at.ac.oeaw.imba.gerlich.looptrace

import cats.data.NonEmptyList
import cats.syntax.all.*
import org.scalacheck.*
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import io.github.iltotore.iron.scalacheck.char.given // for Arbitrary[PositionName]

import at.ac.oeaw.imba.gerlich.gerlib.geometry.Centroid
import at.ac.oeaw.imba.gerlich.gerlib.imaging.*
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.testing.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergerAssessedRoi
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.{
    IndexedDetectedSpot, 
    mergeRois,
}
import at.ac.oeaw.imba.gerlich.looptrace.space.{ BoundingBox, Point3D }

/** Tests for ROI mergers */
class TestMergeRois extends AnyFunSuite, ScalaCheckPropertyChecks, LooptraceSuite, should.Matchers:
    final case class MergerCase(
        adjacencies: Map[RoiIndex, Set[RoiIndex]], 
        expectedSingletons: Set[RoiIndex], 
        expectedContributors: Map[RoiIndex, RoiIndex], // one merge result (value) for each input
        expectedMergers: Map[RoiIndex, Set[RoiIndex]], // one merge result (key) from multiple inputs (values)
    )
    
    object MergerCase:
        def fromRaw(
            adjacencies: Map[Int, Set[Int]],
            expSingle: Set[Int], 
            expContrib: Map[Int, Int], 
            expMerges: Map[Int, Set[Int]],
        ): MergerCase = new MergerCase(
            adjacencies.map{ (k, vs) => RoiIndex.unsafe(k) -> vs.map(RoiIndex.unsafe) }, 
            expSingle.map(RoiIndex.unsafe), 
            expContrib.map{ (k, v) => RoiIndex.unsafe(k) -> RoiIndex.unsafe(v) },
            expMerges.map{ (k, vs) => RoiIndex.unsafe(k) -> vs.map(RoiIndex.unsafe) }
        )

    private def genRoiWithIndex(idx: RoiIndex, partners: Set[RoiIndex])(using 
        Arbitrary[ImagingContext],
        Arbitrary[Centroid[Double]],
        Arbitrary[BoundingBox],
    ): Gen[MergerAssessedRoi] = for
        context <- Arbitrary.arbitrary[ImagingContext]
        center <- Arbitrary.arbitrary[Centroid[Double]]
        box <- Arbitrary.arbitrary[BoundingBox]
    yield MergerAssessedRoi
        .build(
            idx, 
            context, 
            center,
            box,
            partners,
        )
        .leftMap{ msg => new Exception(s"Error generating ROI: $msg") }
        .fold(throw _, identity)

    private def genRois(adj: Map[RoiIndex, Set[RoiIndex]])(using 
        Arbitrary[ImagingContext],
        Arbitrary[Centroid[Double]],
        Arbitrary[BoundingBox],
    ): Gen[List[MergerAssessedRoi]] = 
        Arbitrary.arbitrary[ImagingContext].flatMap{ ctx => 
            given Arbitrary[ImagingContext] = Arbitrary{ Gen.const(ctx) } // Ensure consistent context.
            adj.toList.traverse(genRoiWithIndex.tupled)
        }

    test("Pairwise proximity relations are correctly used to form connected components for ROI merge. #368"):
        val squarePlusTwo = MergerCase.fromRaw(
            Map(
                0 -> Set(), 
                1 -> Set(2, 4), 
                2 -> Set(1, 3), 
                3 -> Set(2, 4), 
                4 -> Set(2, 1),
                5 -> Set(),
            ), 
            Set(0, 5), 
            Set(1, 2, 3, 4).foldLeft(Map()){ (acc, k) => acc + (k -> 6) }, 
            Map(6 -> Set(1, 2, 3, 4))
        )
        
        val hexagon = 
            val inputs = Map(
                0 -> Set(1, 5), 
                1 -> Set(0, 2), 
                2 -> Set(1, 3), 
                3 -> Set(2, 4), 
                4 -> Set(3, 5), 
                5 -> Set(0, 4),
            )
            val mergeIndex = inputs.keySet.max + 1
            MergerCase.fromRaw(
                inputs,
                Set(), 
                inputs.keySet.foldLeft(Map()){ (acc, k) => acc + (k -> mergeIndex) }, 
                Map(mergeIndex -> inputs.keySet)
            )
        
        val takeFirstBox = (_: Point3D, boxes: NonEmptyList[BoundingBox]) => boxes.head
        
        given [A] => Shrink[A] = Shrink.shrinkAny

        forAll (Table("mergerCase", squarePlusTwo, hexagon)) { mergerCase => 
            forAll (genRois(mergerCase.adjacencies)) { rois =>
                // Perform the call under test.
                val (obsSingle, obsContributors, obsMerged) = mergeRois(takeFirstBox)(rois)
                
                /* Check the singletons. */
                obsSingle.length shouldEqual mergerCase.expectedSingletons.size
                obsSingle.map(_.index).toSet shouldEqual mergerCase.expectedSingletons

                /* Check the mergers. */
                obsMerged.length shouldEqual mergerCase.expectedMergers.size
                obsMerged.foldLeft(Map.empty[RoiIndex, Set[RoiIndex]]){ 
                    (acc, roi) => 
                        import at.ac.oeaw.imba.gerlich.gerlib.collections.AtLeast2.syntax.toSet
                        acc + (roi.index -> roi.contributors.toSet)
                } shouldEqual mergerCase.expectedMergers

                /* Check the contributors */
                obsContributors.length shouldEqual mergerCase.expectedContributors.size
                obsContributors
                    .map(roi => roi.index -> roi.mergeOutput)
                    .toMap shouldEqual mergerCase.expectedContributors
            }
        }
end TestMergeRois
