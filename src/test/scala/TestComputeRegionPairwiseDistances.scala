package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.*
import cats.syntax.all.*
import mouse.boolean.*
import squants.space.Nanometers

import org.scalacheck.{Arbitrary, Gen, Shrink}
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.*
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import io.github.iltotore.iron.scalacheck.char.given

import at.ac.oeaw.imba.gerlich.gerlib.SimpleShow
import at.ac.oeaw.imba.gerlich.gerlib.geometry.EuclideanDistance
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.imaging.{
  FieldOfView,
  ImagingChannel,
  ImagingTimepoint,
  PositionName
}
import at.ac.oeaw.imba.gerlich.gerlib.imaging.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.{
  FieldOfViewColumnName,
  TimepointColumnName,
  zCenterColumnName,
  yCenterColumnName,
  xCenterColumnName
}
import at.ac.oeaw.imba.gerlich.gerlib.numeric.*
import at.ac.oeaw.imba.gerlich.gerlib.syntax.all.*

import at.ac.oeaw.imba.gerlich.looptrace.ComputeRegionPairwiseDistances.*
import at.ac.oeaw.imba.gerlich.looptrace.OneBasedFourDigitPositionName.given
import at.ac.oeaw.imba.gerlich.looptrace.collections.*
import at.ac.oeaw.imba.gerlich.looptrace.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.space.*
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*

/** Tests for the simple pairwise distances computation program, for regional
  * barcode spots
  *
  * @author
  *   Vince Reuter
  */
class TestComputeRegionPairwiseDistances
    extends AnyFunSuite,
      ScalaCheckPropertyChecks,
      LooptraceSuite,
      should.Matchers:
  override implicit val generatorDrivenConfig: PropertyCheckConfiguration =
    PropertyCheckConfiguration(minSuccessful = 100)

  val AllReqdColumns = List(
    FieldOfViewColumnName.value,
    TimepointColumnName.value,
    xCenterColumnName[RawCoordinate].value,
    yCenterColumnName[RawCoordinate].value,
    zCenterColumnName[RawCoordinate].value
  )

  private def ToNanometersIdentity: Pixels3D = Pixels3D(
    PixelDefinition.unsafeDefine(Nanometers(1)),
    PixelDefinition.unsafeDefine(Nanometers(1)),
    PixelDefinition.unsafeDefine(Nanometers(1))
  )

  test("Input file that's just a header produces an empty output file.") {
    forAll(Table("includePandasIndexCol", List(false, true)*)) {
      includePandasIndexCol =>
        withTempDirectory { (tempdir: os.Path) =>
          /* Setup and pretests */
          val infile = tempdir / "input.csv"
          val outfolder = tempdir / "output"
          os.makeDir(outfolder)
          val cols =
            if includePandasIndexCol then "" :: AllReqdColumns
            else AllReqdColumns
          os.write(infile, Delimiter.CommaSeparator.join(cols) ++ "\n")
          val expOutfile = outfolder / "input.pairwise_distances__regional.csv"
          os.exists(expOutfile) shouldBe false
          workflow(
            inputFile = infile,
            maybeDriftFile = None,
            pixels = ToNanometersIdentity,
            outputFolder = outfolder
          )
          os.isFile(expOutfile) shouldBe true
          safeReadAllWithOrderedHeaders(expOutfile) match {
            case Left(err) =>
              fail(s"Expected successful output file parse but got error: $err")
            case Right((header, Nil)) => header shouldEqual List()
            case Right((_, records)) =>
              fail(
                s"Expected empty output file but got ${records.length} record(s)!"
              )
          }
        }
    }
  }

  test("Distances computed are accurately Euclidean.") {
    import io.github.iltotore.iron.autoRefine

    def buildPoint(x: Double, y: Double, z: Double) =
      Point3D(XCoordinate(x), YCoordinate(y), ZCoordinate(z))

    val pos = OneBasedFourDigitPositionName
      .fromString(false)("P0001")
      .fold(msg => throw new Exception(msg), identity)
    val channel = ImagingChannel.unsafe(0)
    val inputRecords = NonnegativeInt
      .indexed(List((2.0, 1.0, -1.0), (1.0, 5.0, 0.0), (3.0, 0.0, 2.0)))
      .map { (pt, i) =>
        Input.GoodRecord(
          RoiIndex.unsafe(i),
          pos,
          ImagingTimepoint.unsafe(i),
          channel,
          buildPoint.tupled(pt),
          TraceGroupMaybe.empty
        )
      }
    val expected: Iterable[OutputRecord] = List(0 -> 1, 0 -> 2, 1 -> 2).map {
      (i, j) =>
        val id1 = RoiIndex.unsafe(i)
        val id2 = RoiIndex.unsafe(j)
        val rec1 = inputRecords(i)
        val rec2 = inputRecords(j)
        val t1 = ImagingTimepoint.unsafe(i)
        val t2 = ImagingTimepoint.unsafe(j)
        // TODO: define pixels here
        // val myPixels: Pixels3D = ???
        OutputRecord(
          pos,
          channel,
          t1,
          t2,
          euclideanDistanceBetweenImagePoints(myPixels)(rec1.point, rec2.point),
          id1,
          id2,
          TraceGroupMaybe.empty,
          TraceGroupMaybe.empty
        )
    }
    val observed =
      inputRecordsToOutputRecords(inputRecords, None, ToNanometersIdentity)
    observed shouldEqual expected
  }

  test(
    "When no input records share identical grouping elements, there's never any output."
  ) {
    def genRecords: Gen[NonEmptyList[Input.GoodRecord]] =
      arbitrary[NonEmptyList[Input.GoodRecord]].suchThat { rs =>
        rs.length > 1 &&
        rs.toList.combinations(2).forall {
          case r1 :: r2 :: Nil =>
            Input.getGroupingKey(r1) =!= Input.getGroupingKey(r2)
          case recs =>
            throw new Exception(
              s"Got list of ${recs.length} (not 2) when taking pairs!"
            )
        }
      }

    def genScaling: Gen[Pixels3D] =
      given Arbitrary[PixelDefinition] = Arbitrary {
        Gen.posNum[Double].map { x =>
          PixelDefinition.unsafeDefine(Nanometers(x))
        }
      }
      arbitrary[(PixelDefinition, PixelDefinition, PixelDefinition)]
        .map(Pixels3D.apply)

    forAll(genRecords, genScaling) { (records, pixels) =>
      inputRecordsToOutputRecords(
        records.toList,
        None,
        pixels
      ).isEmpty shouldBe true
    }
  }

  test(
    "Any output record's original record indices map them back to input records with identical grouping key (FOV, channel)."
  ) {
    forAll(
      Gen.choose(10, 100).flatMap(Gen.listOfN(_, arbitrary[Input.GoodRecord])),
      minSuccessful(500)
    ) { (records: List[Input.GoodRecord]) =>
      val indexedRecords = NonnegativeInt.indexed(records)
      val getKey =
        records.map(r => r.index -> r).toMap.apply.andThen(Input.getGroupingKey)
      val observed =
        inputRecordsToOutputRecords(records, None, ToNanometersIdentity)
      observed.filter { r =>
        getKey(r.roiId1) === getKey(r.roiId2)
      } shouldEqual observed
    }
  }

  test("The number of records is always as expected."):
    // To encourage collisions, narrow the choices for grouping components.
    given Arbitrary[PositionName] = Gen
      .oneOf(
        PositionName.unsafe("P0001.zarr"),
        PositionName.unsafe("P0002.zarr")
      )
      .toArbitrary

    // Fix generation to a single constant imaging channel, to make for more colliisions.
    given Arbitrary[ImagingChannel] =
      Gen.const(ImagingChannel.unsafe(0)).toArbitrary

    forAll(
      Gen.choose(5, 50).flatMap(Gen.listOfN(_, arbitrary[Input.GoodRecord])),
      minSuccessful(500)
    ) { (records: List[Input.GoodRecord]) =>
      val indexedRecords = NonnegativeInt.indexed(records)
      val observed =
        inputRecordsToOutputRecords(records, None, ToNanometersIdentity).size
      records
        .map(_.fieldOfView)
        .groupBy(identity)
        .view
        .map(_._2.size)
        .toList
        .toNel match {
        case None => fail("No records!")
        case Some(fovCounts) =>
          val expected = fovCounts.reduceMap { n =>
            (0.5 * n * (n - 1)).toInt
          } // Sum (n choose 2) over each n.
          observed shouldEqual expected
      }
    }

  test(
    "Records are emitted in ascending order according to the composite key: (FOV, region 1, region 2, distance) by function inputRecordsToOutputRecords."
  ):
    /* To encourage collisions, narrow the choices for grouping components. */
    given Arbitrary[PositionName] = Arbitrary {
      Gen.oneOf(
        PositionName.unsafe("P0001.zarr"),
        PositionName.unsafe("P0002.zarr")
      )
    }

    forAll(
      Gen.choose(5, 50).flatMap(Gen.listOfN(_, arbitrary[Input.GoodRecord])),
      minSuccessful(500)
    ) { (records: List[Input.GoodRecord]) =>
      val observed =
        inputRecordsToOutputRecords(records, None, ToNanometersIdentity)
      val expected = observed.toList.sortBy { r =>
        (r.fieldOfView, r.channel, r.timepoint1, r.timepoint2, r.distance)
      }(using
        summon[Order[
          (
              OneBasedFourDigitPositionName,
              ImagingChannel,
              ImagingTimepoint,
              ImagingTimepoint,
              EuclideanDistance
          )
        ]].toOrdering
      )
      observed.toList shouldEqual expected
    }

  test("(FOV, timepoint) is NOT a key!") {
    import io.github.iltotore.iron.autoRefine

    // nCk, i.e. number of ways to choose k indistinguishable objects from n
    def choose(n: Int)(k: Int): Int =
      require(n >= 0 && n <= 10, s"n not in [0, 10] for nCk: $n")
      require(k <= n, s"Cannot choose more items than available: $k > $n")
      val factorial = (z: Int) => (1 to z).product
      factorial(n) / (factorial(k) * factorial(n - k))

    /* To encourage collisions, narrow the choices for grouping components. */
    given Arbitrary[OneBasedFourDigitPositionName] = Gen
      .const(unsafeLiftStringToOneBasedFourDigitPositionName("P0002"))
      .toArbitrary
    given Arbitrary[ImagingChannel] =
      Gen.const(ImagingChannel.unsafe(0)).toArbitrary
    given Arbitrary[ImagingTimepoint] =
      Gen.oneOf(40, 41, 42).map(ImagingTimepoint.unsafe).toArbitrary

    given [A] => Shrink[A] =
      Shrink.shrinkAny // Do absolutely no example shrinking.

    forAll(
      Gen.choose(5, 10).flatMap(Gen.listOfN(_, arbitrary[Input.GoodRecord]))
    ) { (records: List[Input.GoodRecord]) =>
      // Pretest: must be multiple records of same timepoint even within same FOV.
      records
        .groupBy(r => r.fieldOfView -> r.timepoint)
        .view
        .mapValues(_.length)
        .toMap
        .filter(_._2 > 1)
        .nonEmpty shouldBe true
      val getKey = (_: Input.GoodRecord | OutputRecord) match {
        case i: Input.GoodRecord => i.fieldOfView
        case o: OutputRecord     => o.fieldOfView
      }
      val expGroupSizes =
        records.groupBy(getKey).view.mapValues { g => choose(g.size)(2) }.toMap
      val obsGroupSizes =
        inputRecordsToOutputRecords(records, None, ToNanometersIdentity)
          .groupBy(getKey)
          .view
          .mapValues(_.size)
          .toMap
      obsGroupSizes shouldEqual expGroupSizes
    }
  }

  /** Use arbitrary instances for components to derive an an instance for the
    * sum type.
    */
  given (
      arbId: Arbitrary[RoiIndex],
      arbPos: Arbitrary[OneBasedFourDigitPositionName],
      arbTimepoint: Arbitrary[ImagingTimepoint],
      arbChannel: Arbitrary[ImagingChannel],
      arbPoint: Arbitrary[Point3D],
      arbTraceGroup: Arbitrary[TraceGroupMaybe]
  ) => Arbitrary[Input.GoodRecord] =
    (arbId, arbPos, arbTimepoint, arbChannel, arbPoint, arbTraceGroup).mapN(
      Input.GoodRecord.apply
    )
end TestComputeRegionPairwiseDistances
