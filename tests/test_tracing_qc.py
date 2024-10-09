"""Tests for the tracing quality control"""

import pytest
import pandas as pd

from looptrace.tracing_qc_support import compute_ref_timepoint_spatial_information


TRACES_FILE_LINES = """
,position,pos_index,trace_id,timepoint,ref_timepoint,channel,z,y,x
0,P0001.zarr,0,0,0,22,0,-120.7606701885158,-118.6606985242512,-164.68454811375952
1,P0001.zarr,0,0,0,23,0,-120.7606701885158,-118.6606985242512,-164.68454811375952
2,P0002.zarr,1,0,0,22,0,-481.54656448857935,-125.74304582138444,-176.7951017332434
3,P0002.zarr,1,0,0,23,0,-481.54656448857935,-125.74304582138444,-176.7951017332434
60,P0001.zarr,0,0,1,22,0,74632.91518627081,2427.5000894914547,740.2412776217941
61,P0001.zarr,0,0,1,23,0,1669.9270816118196,1992.5648171016517,1976.0165985719548
62,P0002.zarr,1,0,1,22,0,2626.023287785882,1765.484728842217,1418.426687929223
63,P0002.zarr,1,0,1,23,0,2165.692739414378,1966.0277447564467,1626.1823806227096
1320,P0001.zarr,0,0,22,22,0,2213.499389150395,1712.8228077074216,1929.274326035619
1321,P0001.zarr,0,0,22,23,0,1420.8667912166954,2254.4358429387944,686.8115488509881
1322,P0002.zarr,1,0,22,22,0,2951.601577531297,720.9031555705485,708.0171469177135
1323,P0002.zarr,1,0,22,23,0,4430.15058205307,-315.55216037411867,927.9448197386964
1380,P0001.zarr,0,0,23,22,0,-194822.26553040836,1615.4609945787215,1944.6132944898225
1381,P0001.zarr,0,0,23,23,0,1870.510930992522,1695.166448348644,1751.471715525939
1382,P0002.zarr,1,0,23,22,0,2420.5195450391684,2033.652570345409,1764.100670330913
1383,P0002.zarr,1,0,23,23,0,2124.0510990303032,1914.5922427721748,1857.1956666568792
2640,P0001.zarr,0,1,0,22,0,-120.7606701885158,-118.6606985242512,-164.68454811375952
2641,P0001.zarr,0,1,0,23,0,-120.7606701885158,-118.6606985242512,-164.68454811375952
2642,P0002.zarr,1,1,0,22,0,-481.54656448857935,-125.74304582138444,-176.7951017332434
2643,P0002.zarr,1,1,0,23,0,-481.54656448857935,-125.74304582138444,-176.7951017332434
2700,P0001.zarr,0,1,1,22,0,1472.8977202347903,1933.0157037120173,1683.9970995349713
2701,P0001.zarr,0,1,1,23,0,1319.3447784616103,1894.5568546028085,1901.0116259257072
2702,P0002.zarr,1,1,1,22,0,2352.5647965568182,1889.9825394052111,1409.1355184436088
2703,P0002.zarr,1,1,1,23,0,2463.409396516733,1473.358402265271,1609.7558016864448
3960,P0001.zarr,0,1,22,22,0,2040.4853280938587,1709.2109583083902,1715.3366581014786
3961,P0001.zarr,0,1,22,23,0,3630.3897569476767,3418.899952717076,782.0517397783997
3962,P0002.zarr,1,1,22,22,0,2267.948658159447,1961.7971052757389,1492.8708650654469
3963,P0002.zarr,1,1,22,23,0,4290.593152859444,1678.434671019642,944.9148760298984
4020,P0001.zarr,0,1,23,22,0,2263.790871221522,1638.0819327192723,1689.9685458427816
4021,P0001.zarr,0,1,23,23,0,1745.7283054967684,1710.5726367147815,1726.975725006445
4022,P0002.zarr,1,1,23,22,0,2652.6442477667288,1816.3334246086847,1836.4216840136623
4023,P0002.zarr,1,1,23,23,0,2279.7284296761545,1820.0918407943475,1845.3781329337016
""".splitlines(True)

REF_DIST_COL = "ref_dist"
REFERENCE_COORDINATES = ["z_ref", "y_ref", "x_ref"]

REFERENCE_TIMEPOINTS = [22, 23]
NON_REFERENCE_TIMEPOINTS = [0, 1]
HYBRIDISATION_ROUNDS = NON_REFERENCE_TIMEPOINTS + REFERENCE_TIMEPOINTS


@pytest.fixture(scope="session")
def traces_file(tmp_path_factory):
    """Once per test session, write the simplified traces file data."""
    fp = tmp_path_factory.mktemp("data") / "traces.csv"
    with open(fp, 'w') as fh:
        for line in TRACES_FILE_LINES:
            fh.write(line)
    return fp


@pytest.fixture
def traces_table(traces_file):
    """Provide each test case which uses this with a freshly parsed traces table."""
    return pd.read_csv(traces_file)


@pytest.mark.parametrize("coordinate", ["z", "y", "x"])
def test_traces_table_has_coordinates(traces_table, coordinate):
    assert coordinate in traces_table.columns


@pytest.mark.parametrize("reference_coordinate", REFERENCE_COORDINATES)
def test_reference_point_distance_computation_writes_reference__coordindates(traces_table, reference_coordinate):
    assert reference_coordinate not in traces_table.columns
    traces_table = _apply_reference_distances(traces_table)
    assert reference_coordinate in traces_table.columns


@pytest.mark.parametrize("field_of_view", [0, 1])
@pytest.mark.parametrize("trace_id", [0, 1])
@pytest.mark.parametrize("reference_timepoint", REFERENCE_TIMEPOINTS)
@pytest.mark.parametrize("ref_val_col", REFERENCE_COORDINATES)
def test_rows_with_same_fov_region_and_trace_id_have_same_reference_coordinates(traces_table, field_of_view, trace_id, reference_timepoint, ref_val_col):
    traces_table = _apply_reference_distances(traces_table)
    subtab = traces_table[(traces_table.pos_index == field_of_view) & (traces_table.trace_id == trace_id) & (traces_table.ref_timepoint == reference_timepoint)]
    assert 1 == subtab[ref_val_col].nunique()


@pytest.mark.parametrize("ref_val_col", REFERENCE_COORDINATES)
@pytest.mark.parametrize("value_to_check", ["pos_index", "trace_id", "ref_timepoint"])
def test_rows_with_same_reference_coordinates_are_all_same_fov_region_and_trace_id(traces_table, ref_val_col, value_to_check):
    traces_table = _apply_reference_distances(traces_table)
    # In each group, there should be just 1 unique value for the variable of interest.
    ref_val_groups = traces_table.groupby(ref_val_col)[value_to_check].agg(lambda xs: len(set(xs)))
    print(ref_val_groups) # for debugging
    assert (1 == ref_val_groups).all()


@pytest.mark.parametrize("reference_timepoint", REFERENCE_TIMEPOINTS)
def test_all_reference_timepoint_data_points_have_zero_distance(traces_table, reference_timepoint):
    traces_table = _apply_reference_distances(traces_table)
    # Assertion applies only where timepoint is reference timepoint, and we parametrize in each reference timepoint.
    subtab = traces_table[(traces_table.timepoint == traces_table.ref_timepoint) & (traces_table.ref_timepoint == reference_timepoint)]
    print(subtab[['pos_index', 'trace_id', "timepoint", 'ref_timepoint', 'ref_dist']]) # for debugging if failing
    assert not subtab.empty and (subtab[REF_DIST_COL] == 0).all()


def _apply_reference_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the reference points (which stores the coordinates), and compute and add the distance values."""
    ref_dist = compute_ref_timepoint_spatial_information(df)
    df[REF_DIST_COL] = ref_dist
    return df

