"""Tests for determining the spot extraction table, i.e. how to get locus-specific spot data from regional spots"""

from collections.abc import Callable
from enum import Enum
from operator import itemgetter
from typing import Optional

from gertils.types import TimepointFrom0
import hypothesis as hyp
from hypothesis import strategies as st
import more_itertools as more_itools
import pandas as pd
import pytest

from looptrace.SpotPicker import build_locus_spot_data_extraction_table


FOV_NAME = "P0001.zarr"

class BoxSideLengths(Enum):
    """The side lengths of the 3D bounding box, by dimension"""
    Z = 12
    Y = 24
    X = 24

REGIONAL_TIME_1 = 29
REGIONAL_TIME_2 = 30
REGIONAL_TIME_3 = 31
REGIONAL_TIME_4 = 32
REGIONAL_TIME_5 = 33
REGIONAL_TIME_6 = 34
REGIONAL_TIME_7 = 35
REGIONAL_TIME_8 = 36


REGIONAL_SPOT_LINES = f""",position,timepoint,spotChannel,zc,yc,xc,intensityMean,zMin,zMax,yMin,yMax,xMin,xMax
0,{FOV_NAME},{REGIONAL_TIME_1},0,18.177805530982404,445.45646697850475,607.9657421380375,160.33961681087763,12.177805530982404,24.177805530982404,433.45646697850475,457.45646697850475,595.9657421380375,619.9657421380375
1,{FOV_NAME},{REGIONAL_TIME_1},0,17.83959674876146,1006.0753359579252,306.5263466292306,160.1254275940707,11.839596748761458,23.83959674876146,994.0753359579252,1018.0753359579252,294.5263466292306,318.5263466292306
2,{FOV_NAME},{REGIONAL_TIME_2},0,17.70877472362621,1040.482813665982,290.6567022086824,163.12094117647058,11.70877472362621,23.70877472362621,1028.482813665982,1052.482813665982,278.6567022086824,302.6567022086824
3,{FOV_NAME},{REGIONAL_TIME_2},0,17.294947508199503,1701.3423124347908,1665.0076588607058,152.65736930345983,11.294947508199503,23.294947508199503,1689.3423124347908,1713.3423124347908,1653.0076588607058,1677.0076588607058
4,{FOV_NAME},{REGIONAL_TIME_2},0,3.907628987532479,231.9874778925304,871.9833511648726,118.26726920593931,-2.092371012467521,9.90762898753248,219.9874778925304,243.9874778925304,859.9833511648726,883.9833511648726
5,{FOV_NAME},{REGIONAL_TIME_3},0,17.994259347453493,24.042015416774795,1360.0069098862991,117.13946884917321,11.994259347453493,23.994259347453493,12.042015416774795,36.042015416774795,1348.0069098862991,1372.0069098862991
6,{FOV_NAME},{REGIONAL_TIME_3},0,23.009102422189756,231.98008711401275,871.9596645390719,116.14075915047447,17.009102422189756,29.009102422189756,219.98008711401275,243.98008711401275,859.9596645390719,883.9596645390719
7,{FOV_NAME},{REGIONAL_TIME_3},0,16.527137137619988,422.5165732477932,667.2129969610728,157.0530303030303,10.527137137619988,22.527137137619988,410.5165732477932,434.5165732477932,655.2129969610728,679.2129969610728
8,{FOV_NAME},{REGIONAL_TIME_3},0,16.950424004488077,118.88259896330818,349.6540530977019,161.81065410400436,10.950424004488077,22.950424004488077,106.88259896330818,130.88259896330817,337.6540530977019,361.6540530977019
9,{FOV_NAME},{REGIONAL_TIME_4},0,21.068244471265608,342.6729182418883,1638.205658587443,155.96630653726424,15.068244471265608,27.068244471265608,330.6729182418883,354.6729182418883,1626.205658587443,1650.205658587443
10,{FOV_NAME},{REGIONAL_TIME_4},0,21.1040806359985,1676.8233529452802,754.2554990415505,152.48581560283688,15.1040806359985,27.1040806359985,1664.8233529452802,1688.8233529452802,742.2554990415505,766.2554990415505
11,{FOV_NAME},{REGIONAL_TIME_5},0,21.049536953428476,2005.3808438708877,195.2857476439321,152.70525206393816,15.049536953428476,27.049536953428476,1993.3808438708877,2017.3808438708877,183.2857476439321,207.2857476439321
12,{FOV_NAME},{REGIONAL_TIME_5},0,25.89645039712114,150.1274088111918,289.9574239294764,123.1816393442623,19.89645039712114,31.89645039712114,138.1274088111918,162.1274088111918,277.9574239294764,301.9574239294764
13,{FOV_NAME},{REGIONAL_TIME_6},0,23.07234765458799,291.00593975457616,1050.8033122541422,157.6742552676193,17.07234765458799,29.07234765458799,279.00593975457616,303.00593975457616,1038.8033122541422,1062.8033122541422
14,{FOV_NAME},{REGIONAL_TIME_6},0,21.721603405792308,998.9470502825586,1063.7815808196697,157.85837303041407,15.721603405792308,27.721603405792308,986.9470502825586,1010.9470502825586,1051.7815808196697,1075.7815808196697
15,{FOV_NAME},{REGIONAL_TIME_7},0,24.446262903964854,1441.8098380097572,1374.652510713997,175.08730533270412,18.446262903964854,30.446262903964854,1429.8098380097572,1453.8098380097572,1362.652510713997,1386.652510713997
16,{FOV_NAME},{REGIONAL_TIME_7},0,22.97271040371795,1491.5769534466617,1358.6580157020226,183.87555094633134,16.97271040371795,28.97271040371795,1479.5769534466617,1503.5769534466617,1346.6580157020226,1370.6580157020226
17,{FOV_NAME},{REGIONAL_TIME_8},0,29.851626619403987,34.023528359640814,1313.9864553396123,119.01354620222544,23.851626619403987,35.85162661940399,22.023528359640814,46.023528359640814,1301.9864553396123,1325.9864553396123
18,{FOV_NAME},{REGIONAL_TIME_8},0,34.050098785077644,150.12521120339,289.99268722230664,122.25027322404371,28.050098785077644,40.050098785077644,138.12521120339,162.12521120339,277.99268722230664,301.99268722230664
""".splitlines(keepends=True)


DRIFT_CORRECTION_LINES = f""",timepoint,position,zDriftCoarsePixels,yDriftCoarsePixels,xDriftCoarsePixels,zDriftFinePixels,yDriftFinePixels,xDriftFinePixels
0,0,{FOV_NAME},0.0,6.0,-8.0,-0.5830231542670684,-0.531531046456028,0.8461834758622996
1,1,{FOV_NAME},0.0,6.0,-2.0,-0.3410879696406834,-0.4880472193110993,-0.106246765430774
2,2,{FOV_NAME},0.0,4.0,-2.0,-0.2580592154863836,0.4307058830617659,0.1742932564653559
3,3,{FOV_NAME},0.0,4.0,-4.0,-0.2125880036800097,0.2467518090200189,0.8574457677057806
4,4,{FOV_NAME},0.0,4.0,-4.0,-0.2595840366775398,0.616471781253955,0.7317544499834674
5,5,{FOV_NAME},0.0,6.0,-10.0,-0.4958294588115835,-0.2620502768445369,0.2111486770999821
6,6,{FOV_NAME},0.0,8.0,-6.0,-0.4199774166155571,-0.8363321885181345,0.8598764393237204
7,7,{FOV_NAME},0.0,6.0,-6.0,-0.4028533638771586,0.9400429216690104,0.3196463005498426
8,8,{FOV_NAME},0.0,8.0,-12.0,-0.751740275770208,0.0649446340351161,0.7611884980442118
9,9,{FOV_NAME},0.0,8.0,-8.0,-0.4618821337790125,-0.0170220020365881,0.8269084419857513
10,10,{FOV_NAME},0.0,6.0,-6.0,-0.7004585245363046,-0.1994954018115287,0.1585769543406825
11,11,{FOV_NAME},0.0,6.0,-14.0,-0.4831747949593141,0.83436935648359,0.8045305815579386
12,12,{FOV_NAME},0.0,8.0,-10.0,-0.5212184776854787,1.0915660362655928,0.6788574568090504
13,13,{FOV_NAME},0.0,12.0,-16.0,-0.7728099759483871,-0.6390218894193783,-0.022578337281235
14,14,{FOV_NAME},-2.0,12.0,-10.0,1.0285360296476636,-0.57138840628813,-0.2911483984136657
15,15,{FOV_NAME},0.0,8.0,-10.0,-0.8085136043413428,-0.1601949217481472,-0.6535704130133203
16,16,{FOV_NAME},0.0,8.0,-6.0,-0.5104678579845755,-0.105603692409374,-0.6179233982142504
17,17,{FOV_NAME},0.0,8.0,-8.0,-0.8021896396900309,0.0092736230301446,-0.635146941232479
18,18,{FOV_NAME},0.0,10.0,-12.0,-0.528238854114592,-0.4080624089991154,0.8218405339904338
19,19,{FOV_NAME},0.0,10.0,-12.0,-0.8706390984773462,0.0499391191433889,-0.3154285430927288
20,20,{FOV_NAME},0.0,6.0,-4.0,-0.4439357102534834,0.4514788702650442,0.0977879879186602
21,21,{FOV_NAME},0.0,4.0,-10.0,-0.4037546030100427,0.4449851471802941,0.8442647188719049
22,22,{FOV_NAME},0.0,4.0,-6.0,-0.208557238767277,0.4177268471927206,0.8010711034987159
23,23,{FOV_NAME},0.0,4.0,-6.0,-0.44772705514061,-0.3435231071507272,0.2491660273184992
24,24,{FOV_NAME},0.0,4.0,-12.0,-0.5580740676158518,0.5380388604269816,0.5475039943726974
25,25,{FOV_NAME},0.0,6.0,-12.0,-0.5795000340206581,-0.3047916534072151,0.4663888826467513
26,26,{FOV_NAME},0.0,6.0,-6.0,-0.0764284821911207,-0.6652747911096715,0.477891133374911
27,27,{FOV_NAME},0.0,6.0,-12.0,-0.3720103907943052,0.2249002851735589,0.027546619160515
28,28,{FOV_NAME},0.0,8.0,-8.0,-0.3058041481906469,0.2528498900031438,0.1049782332363338
29,{REGIONAL_TIME_1},{FOV_NAME},0.0,6.0,-8.0,-0.2864363341913231,0.9334388629806376,-0.7219706089158727
30,{REGIONAL_TIME_2},{FOV_NAME},0.0,8.0,-14.0,-0.2033370146070154,-0.4396433120547883,-0.7750550866106661
31,{REGIONAL_TIME_3},{FOV_NAME},0.0,10.0,-12.0,-0.1779836033029637,0.9596877685781467,0.4481122560268285
32,{REGIONAL_TIME_4},{FOV_NAME},0.0,10.0,-12.0,-0.4195241432323118,0.9314775021773122,0.0304283433778342
33,{REGIONAL_TIME_5},{FOV_NAME},0.0,10.0,-18.0,-0.5680507925624895,0.1492476459938268,0.7950874541064629
34,{REGIONAL_TIME_6},{FOV_NAME},0.0,12.0,-18.0,-0.4043858880352026,0.3456730380175369,0.0732578815987924
35,{REGIONAL_TIME_7},{FOV_NAME},0.0,14.0,-20.0,-0.4892526435283817,0.8451213846188307,0.657246600130344
36,{REGIONAL_TIME_8},{FOV_NAME},0.0,16.0,-20.0,-0.7560933247411967,0.3851487162033655,0.1825965588127672
""".splitlines(keepends=True)


ROUNDS_CONFIG_BASE_DATA = {
    "imagingRounds": [
        {"time": 0, "name": "pre_image", "isBlank": True},
        {"time": 1, "probe": "Dp001"},
        {"time": 2, "probe": "Dp002"},
        {"time": 3, "probe": "Dp003"},
        {"time": 4, "probe": "Dp006"},
        {"time": 5, "probe": "Dp007"},
        {"time": 6, "probe": "Dp009"},
        {"time": 7, "probe": "Dp010"},
        {"time": 8, "probe": "Dp011"},
        {"time": 9, "probe": "Dp012"},
        {"time": 10, "probe": "Dp125"},
        {"time": 11, "probe": "Dp128"},
        {"time": 12, "probe": "Dp129"},
        {"time": 13, "probe": "Dp130"},
        {"time": 14, "probe": "Dp131"},
        {"time": 15, "probe": "Dp132"},
        {"time": 16, "probe": "Dp133"},
        {"time": 17, "probe": "Dp136"},
        {"time": 18, "probe": "Dp137"},
        {"time": 19, "probe": "Dp138"},
        {"time": 20, "probe": "Dp139"},
        {"time": 21, "probe": "Dp140"},
        {"time": 22, "probe": "Dp141"},
        {"time": 23, "probe": "Dp001", "repeat": 1},
        {"time": 24, "name": "blank_01", "isBlank": True},
        {"time": 25, "probe": "Dp148"},
        {"time": 26, "probe": "Dp149"},
        {"time": 27, "probe": "Dp150"},
        {"time": 28, "name": "blank_02", "isBlank": True},
        {"time": REGIONAL_TIME_1, "probe": "Dp102", "isRegional": True},
        {"time": REGIONAL_TIME_2, "probe": "Dp142", "isRegional": True},
        {"time": REGIONAL_TIME_3, "probe": "Dp107", "isRegional": True},
        {"time": REGIONAL_TIME_4, "probe": "Dp144", "isRegional": True},
        {"time": REGIONAL_TIME_5, "probe": "Dp105", "isRegional": True},
        {"time": REGIONAL_TIME_6, "probe": "Dp143", "isRegional": True},
        {"time": REGIONAL_TIME_7, "probe": "Dp109", "isRegional": True},
        {"time": REGIONAL_TIME_8, "probe": "Dp146", "isRegional": True},
    ],
}


NON_REGIONAL_ROUNDS, REGIONAL_ROUNDS = more_itools.partition(
    lambda r: r.get("isRegional", False) is True, 
    ROUNDS_CONFIG_BASE_DATA["imagingRounds"],
)
REGIONAL_TIMES = tuple(r["time"] for r in REGIONAL_ROUNDS)
NON_REGIONAL_TIMES = tuple(r["time"] for r in NON_REGIONAL_ROUNDS)


@hyp.given(
    locus_grouping=st.one_of(
        st.dictionaries(
            keys=st.sampled_from(REGIONAL_TIMES).map(TimepointFrom0), 
            values=st.sets(st.sampled_from(NON_REGIONAL_TIMES).map(TimepointFrom0)),
            max_size=len(REGIONAL_TIMES),
        ),
        st.just(None),
    )
)
@hyp.settings(
    deadline=None,
    phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink), # Save test execution time.
    suppress_health_check=(hyp.HealthCheck.function_scoped_fixture, ), # We overwrite the files each time, so all good.
)
def test_only_region_timepoints_and_their_locus_timepoints_have_records_in_spot_extraction_table(
    tmp_path, 
    locus_grouping,
):
    
    # First, construct the drift correction table...
    drift_file = tmp_path / "drift.csv"
    with drift_file.open(mode="w") as fh:
        for drift_line in DRIFT_CORRECTION_LINES:
            fh.write(drift_line)
    drift_table: pd.DataFrame = pd.read_csv(drift_file, index_col=0)
    # Subtract 1 to account for header.
    assert drift_table.shape[0] == len(DRIFT_CORRECTION_LINES) - 1, f"{drift_table.shape[0]} row(s) in DC table, from {len(DRIFT_CORRECTION_LINES)} record line(s); these should match."

    # ...then, construct the regional spots table...
    rois_file = tmp_path / "rois.csv"
    with rois_file.open(mode="w") as fh:
        for roi_line in REGIONAL_SPOT_LINES:
            fh.write(roi_line)
    rois_table: pd.DataFrame = pd.read_csv(rois_file, index_col=0)

    # ...then, check that downstream assumption of exactly 1 FOV in each data table is valid.
    assert rois_table["position"].nunique() == 1, f"Expected just 1 unique position in ROIs table, but got {rois_table.position.nunique()}"
    assert drift_table["position"].nunique() == 1, f"Expected just 1 unique position in drift table, but got {drift_table.position.nunique()}"

    exp_region_times: set[int]
    exp_reg_time_loc_time_pairs: list[(int, int)]
    get_locus_timepoints: Optional[Callable[[TimepointFrom0], set[TimepointFrom0]]]
    if not locus_grouping:
        get_locus_timepoints = None
        exp_region_times = set(REGIONAL_TIMES)
        exp_reg_time_loc_time_pairs = [
            (rt, lt) 
            for rt in rois_table["timepoint"]
            for lt in sorted(REGIONAL_TIMES + NON_REGIONAL_TIMES)
        ]
    else:
        get_locus_timepoints = lambda t: locus_grouping.get(t, set())
        exp_region_times: set[int] = set()
        exp_reg_time_loc_time_pairs = []
        # NB: Assume the above input ROI lines are well ordered (this is how real input should be).
        for rt, loc_times in sorted(locus_grouping.items(), key=itemgetter(0)):
            if not loc_times:
                continue
            exp_region_times.add(rt.get)
            # The regional spot itself is collected w/ its locus spots.
            loc_times = sorted([rt, *loc_times])
            # One record is generated per locus timepoint, per regional spot.
            curr_reg_time_spot_count = (rois_table["timepoint"] == rt.get).sum()
            exp_reg_time_loc_time_pairs.extend([(rt.get, lt.get) for lt in loc_times] * curr_reg_time_spot_count)

    spot_extraction_table: pd.DataFrame = build_locus_spot_data_extraction_table(
        rois_table=rois_table,
        get_pos_idx=lambda _: 0,
        get_dc_table=lambda _: drift_table[drift_table["position"] == FOV_NAME], # There's only 1 FOV, so return the whole table, always.
        get_locus_timepoints=get_locus_timepoints,
        get_zyx=lambda _1, _2: (BoxSideLengths.Z.value, BoxSideLengths.Y.value, BoxSideLengths.X.value)
    )

    print(f"Obs row count: {spot_extraction_table.shape[0]}")

    obs_reg_time_loc_time_pairs: list[(int, int)] = \
        [(row["ref_timepoint"], row["timepoint"]) for _, row in spot_extraction_table.iterrows()]
    
    obs_region_times = set(spot_extraction_table["ref_timepoint"].to_list())
    assert obs_region_times == exp_region_times, f"Expected and Observed region times differ: Expected = {exp_region_times}. Observed = {obs_region_times}"

    assert len(obs_reg_time_loc_time_pairs) == len(exp_reg_time_loc_time_pairs)
    assert obs_reg_time_loc_time_pairs == exp_reg_time_loc_time_pairs


@pytest.mark.skip("not yet implemented")
def test_spot_table_construction_stores_line_number_as_roi_number_and_row_index_as_roi_id():
    pass
