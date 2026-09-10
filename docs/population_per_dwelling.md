# Updating population per dwelling data, based on census data in Beyond2020
From the census files, some counts are available as population per household instead of population per dwelling. Population per household counts are generally lower than population per dwelling, as on average there is slightly more then one household per dwelling (e.g., 1.08). For pre-1961, using person per household and household per dwelling to estimate values could be an interesting approach (cf data in scenarios).

The population per dwelling counts vary with urban/rural and owned/rented context. For instance, in 1961, the total average across all dwellings varies from 4.0-6.0 based on urban/rural, and plus/minus 1.0 based on tenure (owned/rented).

PpD = {
    2021: {  # cf 98100041; weights= total hhld size (==dwlgs)
        "total": 2.2,
        "apartments": np.average(
            [2.1, 1.8, 1.6], weights=[271_240, 1_242_910, 225_750]
        ),
        "mobile": 1.9,
        "single_attached": np.average(
            [2.5, 2.5, 2.1], weights=[199_085, 98_630, 15_745]
        ),
        "single_detached": 2.6,
    },
    2016: {  # cf 98400X2016017
        "total": 2.3,
        "apartments": np.average(
            [2.1, 1.8, 1.6], weights=[265_195, 1_154_950, 187_795]
        ),
        "mobile": 1.9,
        "single_attached": np.average(
            [2.6, 2.5, 2.1], weights=[188_245, 93_355, 15_940]
        ),
        "single_detached": 2.6,
    },
    2011: {  # cf 98313XCB2011023
        "total": 2.3,
        "apartments": np.average(
            [2.1, 1.8, 1.6], weights=[263_860, 1_103_845, 171_110]
        ),
        "mobile": 2.0,
        "single_attached": np.average(
            [2.6, 2.4, 2.1], weights=[171_435, 86_040, 15_650]
        ),
        "single_detached": 2.7,
    },
    2006: {  # cf 97554XCB2006032
        "total": 2.3,
        "apartments": np.average(
            [2.2, 1.9, 1.6], weights=[255_925, 1_045_815, 162_270]
        ),
        "mobile": 2.1,
        "single_attached": np.average(
            [2.6, 2.5, 2.2], weights=[154_730, 76_300, 15_775]
        ),
        "single_detached": 2.7,
    },
    2001: {  # cf 95F0327XCB01006
        "total": 2.4,
        "apartments": np.average(
            [2.1, 1.9, 1.6], weights=[155_345, 1_033_275, 154_225]
        ),
        "mobile": 2.3,
        "single_attached": np.average(
            [2.7, 2.5, 2.1], weights=[144_440, 79_795, 19_175]
        ),
        "single_detached": 2.8,
    },
    1996: {  # cf 95F0200XCB96001
        "total": 2.5,
        "apartments": np.average(
            [1.6, 2.1], weights=[144_780, 1_385_295]
        ),  # inferred, see note
        "mobile": 2.4,
        "single_attached": np.nan,  # 2.1 for 'other dwelling'; see note
        "single_detached": 3.0,
    },  # only avail. apart>5 (1.6) and other (2.1). the average of 2.05 may slightly overestimate the actual value for appartments, and underestimate the value for single attached. Overall, the weighted average works: ``np.average([2.7, 2.5, 2.1], weights=[144_440, 79_795, 19_175])``
    1991: {  # cf 1001211 (H9101)
        "total": 2.6,
        "apartments": np.average(
            [1.6, 2.2], weights=[137_105, 1_297_385]
        ),  # inferred, see note
        "mobile": 2.6,
        "single_attached": np.nan,  # 2.2 for 'other dwelling'; see note
        "single_detached": 3.0,
    },  # only avail. apart>5 (1.6) and other (2.2); see comments for 1996
    # From census pdf
    1986: {  # voir analyse csddw86a02; ref CS93-104-1987
        "total": 2.7,
        "apartments": np.average(
            [1.6, 2.4], weights=[116_110, 1_191_455]
        ),  # inferred, see comments 1996
        "mobile": 2.8,
        "single_attached": np.nan,  # 2.4 for 'other dwelling'; see note
        "single_detached": 3.2,
    },  # only avail. apart>5 (1.6) and other (2.4); see comments 1996
    1981: {  # cf 1981929031982engfra.pdf, T5-1
        "total": 2.9,
        "apartments": np.average(
            [2.8, 2.3, 1.7], weights=[239_190, 597_990, 115_515]
        ),  # includes duplex
        "mobile": 3.0,
        "single_attached": 3.0,
        "single_detached": 3.5,
    },  # only avail. apart>5 (1.7) and other (2.5)
    1976: {  # cf 1976938031978engfra.pdf T9
        "total": 3.2,
        "apartments": np.average(
            [2.6, 3.1], weights=[847_430, 149_995]
        ),  # apartments, duplex
        "mobile": 3.0,
        "single_attached": 3.5,
        "single_detached": 3.9,
    },
    1971: {  # cf 1971937381975engfra
        "total": 3.7,
        "apartments": 3.0,
        "mobile": np.nan,  # NA
        "single_attached": 3.6,
        "single_detached": 4.4,
    },  # also avail., rooms/bedrooms per dwlg
    1966: {  # cs93-607-1966  moyenne de personnes par ménage, généralement plus faible que personnes / maison
        "total": 4.0,
        "apartments": 3.4,
        "mobile": np.nan,  # NA,
        "single_attached": 4.0,
        "single_detached": 4.8,
    },
    1961: {  # cs93-529-1961  moyenne de personnes par ménage, généralement plus faible que personnes / maison
        "total": 4.2,
        "apartments": 3.7,
        "mobile": np.nan,  # NA,
        "single_attached": 4.3,
        "single_detached": 5.1,
    },
    1956: {  # cs98-1956-1 T.34  personnes par ménage, généralement plus faible que personnes / maison
        "total": 4.4,
        "apartments": np.nan,  # NA,
        "mobile": np.nan,  # NA,
        "single_attached": np.nan,  # NA,
        "single_detached": np.nan,
    },  # NA,
    1951: {  # cs98-1951-3 T.1-1
        "total": 4.7,
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1941: {  # cs98-1951-3 T.1-1
        "total": 5.17,
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1931: {  # cs98-1951-3 T.1-1
        "total": 5.36,
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1921: {  # cs98-1951-3 T.1-1
        "total": 5.93,
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1911: {  # cs98-1951-3 T.1-1
        "total": 5.90,
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1901: {  # cs98-1951-3 T.1-1
        "total": 5.66,
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1891: {  # cs98-1951-3 T.1-1
        "total": 6.04,
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1881: {  # cs98-1951-3 T.1-1
        "total": 6.28,
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1871: {  # cs98-1608-1876-5 p.54
        # relèvent également que 6.6 pers/maison, 5.6 pers/ménage
        # retrouver la source ds cs1871, mais 148.02 maisons / 1000 pers.
        # même ordre de grandeur
        "total": 1191516 / 180615,  # maisons habitées, Québec
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1861: {  # cs98-1665-1871-4 P.370; aussi cs98-1608-1876-5 p.52
        # relèvent également que 7.2 pers/maison, 6.0 pers/ménage
        "total": 1111566 / 155088,  # maisons habitées, bas canada
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1851: {  # cs98-1665-1871-4 P.292; aussi cs98-1608-1876-5 p.50
        # relèvent également que 7.1 pers/maison, 6.2 pers/ménage
        "total": 890261 / 123983,  # total habitées, bas canada
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1844: {  # cs98-1665-1871-4 P.234
        "total": 697084 / 108749,  # demeures habitées, bas canada
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1831: {  # cs98-1665-1871-4 P.196
        "total": 553134 / 82437,  # demeures habitées, bas canada
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1784: {  # cs98-1665-1871-4 P.164
        "total": 113012 / 18924,  # maisons, 'Canada' (Qc, TR, Mtl)
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1765: {  # cs98-1665-1871-4 P.157
        "total": 69810 / 12230,  # maisons, 'Canada'
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1698: {  # cs98-1665-1871-4 P.130
        "total": 15355 / 2310,  # maisons, nouvelle-france
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1695: {  # cs98-1665-1871-4 P.123
        "total": 13639 / 1934,  # maisons, nouvelle-france
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1692: {  # cs98-1665-1871-4 P.119
        "total": 12431 / 1929,  # maisons + cabanes, nouvelle-france
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1688: {  # cs98-1665-1871-4 P.112
        "total": 11562 / 1877,  # maisons + cabanes, nouvelle-france
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
    1685: {  # cs98-1665-1871-4 P.106
        "total": 12263 / 1990,  # maisons + cabanes, nouvelle-france
        "apartments": np.nan,
        "mobile": np.nan,
        "single_attached": np.nan,
        "single_detached": np.nan,
    },
}

To save as jsonl: ppd.reset_index(names="year").to_json("pop_per_dwlg.jsonl", orient='records', lines=True)