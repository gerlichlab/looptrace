"""Tests for determining the spot extraction table, i.e. how to get locus-specific spot data from regional spots"""

from collections.abc import Callable
from enum import Enum
from operator import itemgetter
from typing import Optional

import hypothesis as hyp
from hypothesis import strategies as st
import more_itertools as more_itools
import pandas as pd

from gertils.types import TimepointFrom0

from looptrace.SpotPicker import build_locus_spot_data_extraction_table

from tests.hypothesis_extra_strategies import gen_locus_grouping_data, gen_proximity_filter_strategy


FOV_NAME = "P0001.zarr"

class BoxSideLengths(Enum):
    """The side lengths of the 3D bounding box, by dimension"""
    Z = 12
    Y = 24
    X = 24

REGIONAL_TIME_1 = 79
REGIONAL_TIME_2 = 80
REGIONAL_TIME_3 = 81
REGIONAL_TIME_4 = 82
REGIONAL_TIME_5 = 83
REGIONAL_TIME_6 = 84
REGIONAL_TIME_7 = 85
REGIONAL_TIME_8 = 86


REGIONAL_SPOT_LINES = f""",position,frame,ch,zc,yc,xc,intensity_mean,z_min,z_max,y_min,y_max,x_min,x_max
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


DRIFT_CORRECTION_LINES = f""",frame,position,z_px_coarse,y_px_coarse,x_px_coarse,z_px_fine,y_px_fine,x_px_fine
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
10,10,{FOV_NAME},0.0,8.0,-14.0,-0.6838701307013478,0.4741362492226904,1.0422407063593535
11,11,{FOV_NAME},0.0,12.0,-8.0,-0.7891703629517751,-0.461303256604174,0.3813733996438314
12,12,{FOV_NAME},0.0,8.0,-4.0,-0.6651601010295104,-0.2260973572273882,-0.6586451879170494
13,13,{FOV_NAME},0.0,6.0,-8.0,-0.6178388634763504,-0.1680046707122089,-0.6400992371922569
14,14,{FOV_NAME},0.0,6.0,-6.0,-0.6102710787342098,1.053306291632592,1.037545676506513
15,15,{FOV_NAME},0.0,6.0,-6.0,-0.3359586640707249,-0.0233705090006067,0.5666842242980821
16,16,{FOV_NAME},0.0,8.0,-6.0,-0.5580035998808041,0.102124683531911,-0.7507902860348489
17,17,{FOV_NAME},0.0,10.0,-8.0,-0.4970709151574861,-0.7642109296763099,-0.7455805271310504
18,18,{FOV_NAME},0.0,10.0,-8.0,-0.5565485747916656,-0.8578507115167348,0.0414963970870132
19,19,{FOV_NAME},0.0,8.0,-8.0,-0.4815732938763827,-0.1868531451154954,-0.4421895045218642
20,20,{FOV_NAME},0.0,8.0,-10.0,-0.4431708590451679,0.3788962261995859,0.794662659082395
21,21,{FOV_NAME},0.0,10.0,-16.0,-0.6230809161238402,0.5001230701185956,0.6173742806997318
22,22,{FOV_NAME},0.0,12.0,-14.0,-0.8826062376305228,0.7243250573317911,-0.3374992730636495
23,23,{FOV_NAME},0.0,10.0,-8.0,-0.5441624985472593,-0.480306271790518,0.945057217452388
24,24,{FOV_NAME},0.0,4.0,-6.0,-0.2928004213711535,0.2081249718642011,0.7979422805376092
25,25,{FOV_NAME},0.0,6.0,-10.0,-0.451091726289784,-0.2952788502543024,0.0616460772712224
26,26,{FOV_NAME},0.0,6.0,-6.0,-0.2313811955411846,0.6472286072434117,0.1944860104437647
27,27,{FOV_NAME},0.0,8.0,-12.0,-0.516872976026765,-0.8052577353032424,0.2546496709644028
28,28,{FOV_NAME},0.0,10.0,-12.0,-0.7498550721761594,-0.4821690491824386,-0.5796817864757995
29,29,{FOV_NAME},0.0,8.0,-4.0,-0.7182070399282755,0.3469548402877105,-0.3743648080444866
30,30,{FOV_NAME},0.0,6.0,-4.0,-0.3661989557696192,-0.3737570523431334,-0.7260201559984452
31,31,{FOV_NAME},0.0,4.0,-10.0,-0.4080127963122935,0.8382434202315059,0.3619589526243266
32,32,{FOV_NAME},0.0,6.0,-6.0,-0.5552258245879323,0.8364576110312945,0.2727650261701501
33,33,{FOV_NAME},0.0,8.0,-12.0,-0.8256879218686056,0.5181866517084566,1.150225514722805
34,34,{FOV_NAME},0.0,10.0,-10.0,-0.8733263245163347,-0.5220866755076645,0.292817444490085
35,35,{FOV_NAME},0.0,6.0,0.0,-0.5280475337147186,-0.3685182568465661,-0.3141820310468779
36,36,{FOV_NAME},0.0,0.0,0.0,0.0,0.0,0.0
37,37,{FOV_NAME},0.0,0.0,-2.0,-0.3917201610729306,0.1606765745727933,0.8234016619488823
38,38,{FOV_NAME},0.0,2.0,-4.0,-0.2619926624435965,-0.7806540710753214,0.8702665421782053
39,39,{FOV_NAME},0.0,4.0,-8.0,-0.3911684407199248,-0.8629997694035036,0.3675222786192995
40,40,{FOV_NAME},0.0,4.0,-4.0,-0.367357634742094,0.322465577821586,-0.320590337479053
41,41,{FOV_NAME},0.0,4.0,-10.0,-0.5470768437392161,1.0507378032129266,0.0739923474013617
42,42,{FOV_NAME},0.0,6.0,-12.0,-0.6125974023587675,0.4009707935685781,0.6927278323651902
43,43,{FOV_NAME},0.0,8.0,-8.0,-0.5041317764277848,0.3545424411807482,-0.1671529395538546
44,44,{FOV_NAME},0.0,8.0,-10.0,-0.4340610441245839,-0.3637573821318143,0.2200223498431711
45,45,{FOV_NAME},0.0,8.0,-10.0,-0.6126466653366904,-0.498353176951702,-0.2766782857538283
46,46,{FOV_NAME},0.0,8.0,-12.0,-0.7469388637804877,0.4778481533202428,0.3025146557675872
47,47,{FOV_NAME},0.0,10.0,-18.0,-0.8016990774805877,-0.7043485658794569,1.0394720210522834
48,48,{FOV_NAME},0.0,8.0,-10.0,-0.6359691298781847,0.4399546700469811,1.1214596108268031
49,49,{FOV_NAME},0.0,4.0,-10.0,-0.329483046156666,0.6293382635962277,0.1027500013721933
50,50,{FOV_NAME},0.0,6.0,-6.0,-0.3291696103632162,0.0124922026008573,-0.4758695123070646
51,51,{FOV_NAME},0.0,6.0,-8.0,-0.5439477633569676,-0.6613669487028531,-0.7285320901715039
52,52,{FOV_NAME},0.0,8.0,-16.0,-0.7154201811998465,-0.6010788622420085,0.8291061417689681
53,53,{FOV_NAME},0.0,10.0,-14.0,-0.651290145305247,0.6074527941907527,0.3008936972495368
54,54,{FOV_NAME},0.0,10.0,-10.0,-0.4272293494158144,-0.7788171223194169,0.1040013839965675
55,55,{FOV_NAME},0.0,8.0,-16.0,-0.7516390817736306,0.3274451178441104,0.16891968363636
56,56,{FOV_NAME},0.0,10.0,-12.0,-0.764557778464685,0.5715005897861373,-0.0047403538219354
57,57,{FOV_NAME},0.0,12.0,-18.0,-0.7561239599328127,-0.309687775219695,0.2607895800453738
58,58,{FOV_NAME},0.0,14.0,-18.0,-0.9115863047696022,-0.2731833054926771,0.319533242948736
59,59,{FOV_NAME},0.0,12.0,-10.0,-0.8059931997227946,0.5442204592506144,0.6099310084287922
60,60,{FOV_NAME},0.0,6.0,-6.0,-0.7004585245363046,-0.1994954018115287,0.1585769543406825
61,61,{FOV_NAME},0.0,6.0,-14.0,-0.4831747949593141,0.83436935648359,0.8045305815579386
62,62,{FOV_NAME},0.0,8.0,-10.0,-0.5212184776854787,1.0915660362655928,0.6788574568090504
63,63,{FOV_NAME},0.0,12.0,-16.0,-0.7728099759483871,-0.6390218894193783,-0.022578337281235
64,64,{FOV_NAME},-2.0,12.0,-10.0,1.0285360296476636,-0.57138840628813,-0.2911483984136657
65,65,{FOV_NAME},0.0,8.0,-10.0,-0.8085136043413428,-0.1601949217481472,-0.6535704130133203
66,66,{FOV_NAME},0.0,8.0,-6.0,-0.5104678579845755,-0.105603692409374,-0.6179233982142504
67,67,{FOV_NAME},0.0,8.0,-8.0,-0.8021896396900309,0.0092736230301446,-0.635146941232479
68,68,{FOV_NAME},0.0,10.0,-12.0,-0.528238854114592,-0.4080624089991154,0.8218405339904338
69,69,{FOV_NAME},0.0,10.0,-12.0,-0.8706390984773462,0.0499391191433889,-0.3154285430927288
70,70,{FOV_NAME},0.0,6.0,-4.0,-0.4439357102534834,0.4514788702650442,0.0977879879186602
71,71,{FOV_NAME},0.0,4.0,-10.0,-0.4037546030100427,0.4449851471802941,0.8442647188719049
72,72,{FOV_NAME},0.0,4.0,-6.0,-0.208557238767277,0.4177268471927206,0.8010711034987159
73,73,{FOV_NAME},0.0,4.0,-6.0,-0.44772705514061,-0.3435231071507272,0.2491660273184992
74,74,{FOV_NAME},0.0,4.0,-12.0,-0.5580740676158518,0.5380388604269816,0.5475039943726974
75,75,{FOV_NAME},0.0,6.0,-12.0,-0.5795000340206581,-0.3047916534072151,0.4663888826467513
76,76,{FOV_NAME},0.0,6.0,-6.0,-0.0764284821911207,-0.6652747911096715,0.477891133374911
77,77,{FOV_NAME},0.0,6.0,-12.0,-0.3720103907943052,0.2249002851735589,0.027546619160515
78,78,{FOV_NAME},0.0,8.0,-8.0,-0.3058041481906469,0.2528498900031438,0.1049782332363338
79,{REGIONAL_TIME_1},{FOV_NAME},0.0,6.0,-8.0,-0.2864363341913231,0.9334388629806376,-0.7219706089158727
80,{REGIONAL_TIME_2},{FOV_NAME},0.0,8.0,-14.0,-0.2033370146070154,-0.4396433120547883,-0.7750550866106661
81,{REGIONAL_TIME_3},{FOV_NAME},0.0,10.0,-12.0,-0.1779836033029637,0.9596877685781467,0.4481122560268285
82,{REGIONAL_TIME_4},{FOV_NAME},0.0,10.0,-12.0,-0.4195241432323118,0.9314775021773122,0.0304283433778342
83,{REGIONAL_TIME_5},{FOV_NAME},0.0,10.0,-18.0,-0.5680507925624895,0.1492476459938268,0.7950874541064629
84,{REGIONAL_TIME_6},{FOV_NAME},0.0,12.0,-18.0,-0.4043858880352026,0.3456730380175369,0.0732578815987924
85,{REGIONAL_TIME_7},{FOV_NAME},0.0,14.0,-20.0,-0.4892526435283817,0.8451213846188307,0.657246600130344
86,{REGIONAL_TIME_8},{FOV_NAME},0.0,16.0,-20.0,-0.7560933247411967,0.3851487162033655,0.1825965588127672
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
        {"time": 10, "probe": "Dp013"},
        {"time": 11, "probe": "Dp014"},
        {"time": 12, "probe": "Dp015"},
        {"time": 13, "probe": "Dp017"},
        {"time": 14, "probe": "Dp018"},
        {"time": 15, "probe": "Dp020"},
        {"time": 16, "probe": "Dp021"},
        {"time": 17, "probe": "Dp025"},
        {"time": 18, "probe": "Dp027"},
        {"time": 19, "probe": "Dp028"},
        {"time": 20, "probe": "Dp032"},
        {"time": 21, "probe": "Dp033"},
        {"time": 22, "probe": "Dp035"},
        {"time": 23, "probe": "Dp036"},
        {"time": 24, "probe": "Dp038"},
        {"time": 25, "probe": "Dp040"},
        {"time": 26, "probe": "Dp041"},
        {"time": 27, "probe": "Dp042"},
        {"time": 28, "probe": "Dp043"},
        {"time": 29, "probe": "Dp045"},
        {"time": 30, "probe": "Dp046"},
        {"time": 31, "probe": "Dp047"},
        {"time": 32, "probe": "Dp049"},
        {"time": 33, "probe": "Dp050"},
        {"time": 34, "probe": "Dp051"},
        {"time": 35, "probe": "Dp054"},
        {"time": 36, "probe": "Dp057"},
        {"time": 37, "probe": "Dp058"},
        {"time": 38, "probe": "Dp059"},
        {"time": 39, "probe": "Dp060"},
        {"time": 40, "probe": "Dp063"},
        {"time": 41, "probe": "Dp067"},
        {"time": 42, "probe": "Dp068"},
        {"time": 43, "probe": "Dp070"},
        {"time": 44, "probe": "Dp071"},
        {"time": 45, "probe": "Dp072"},
        {"time": 46, "probe": "Dp074"},
        {"time": 47, "probe": "Dp080"},
        {"time": 48, "probe": "Dp083"},
        {"time": 49, "probe": "Dp085"},
        {"time": 50, "probe": "Dp086"},
        {"time": 51, "probe": "Dp087"},
        {"time": 52, "probe": "Dp089"},
        {"time": 53, "probe": "Dp096"},
        {"time": 54, "probe": "Dp098"},
        {"time": 55, "probe": "Dp099"},
        {"time": 56, "probe": "Dp100"},
        {"time": 57, "probe": "Dp114"},
        {"time": 58, "probe": "Dp119"},
        {"time": 59, "probe": "Dp123"},
        {"time": 60, "probe": "Dp125"},
        {"time": 61, "probe": "Dp128"},
        {"time": 62, "probe": "Dp129"},
        {"time": 63, "probe": "Dp130"},
        {"time": 64, "probe": "Dp131"},
        {"time": 65, "probe": "Dp132"},
        {"time": 66, "probe": "Dp133"},
        {"time": 67, "probe": "Dp136"},
        {"time": 68, "probe": "Dp137"},
        {"time": 69, "probe": "Dp138"},
        {"time": 70, "probe": "Dp139"},
        {"time": 71, "probe": "Dp140"},
        {"time": 72, "probe": "Dp141"},
        {"time": 73, "probe": "Dp001", "repeat": 1},
        {"time": 74, "name": "blank_01", "isBlank": True},
        {"time": 75, "probe": "Dp148"},
        {"time": 76, "probe": "Dp149"},
        {"time": 77, "probe": "Dp150"},
        {"time": 78, "name": "blank_02", "isBlank": True},
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
        gen_locus_grouping_data.with_strategies_and_empty_flag(
            gen_raw_reg_time=st.sampled_from(REGIONAL_TIMES), 
            gen_raw_loc_time=st.sampled_from(NON_REGIONAL_TIMES),
            max_size=len(REGIONAL_TIMES),
            allow_empty=True,
        ),
        st.just(None),
    )
)
@hyp.settings(
    suppress_health_check=(hyp.HealthCheck.function_scoped_fixture, ), # We overwrite the files each time, so all good.
    phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink), # Save test execution time.
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
            for rt in rois_table["frame"]
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
            curr_reg_time_spot_count = (rois_table["frame"] == rt.get).sum()
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
        [(row["ref_frame"], row["frame"]) for _, row in spot_extraction_table.iterrows()]
    
    obs_region_times = set(spot_extraction_table["ref_frame"].to_list())
    assert obs_region_times == exp_region_times, f"Expected and Observed region times differ: Expected = {exp_region_times}. Observed = {obs_region_times}"

    assert len(obs_reg_time_loc_time_pairs) == len(exp_reg_time_loc_time_pairs)
    assert obs_reg_time_loc_time_pairs == exp_reg_time_loc_time_pairs
