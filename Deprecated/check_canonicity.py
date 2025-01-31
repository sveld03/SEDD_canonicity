from transformers import GPT2TokenizerFast
import torch

torch.set_printoptions(threshold=10000)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def check_canonicity(actual_tokens, canonical_tokens):
    tokens_matched = 0
    total_tokens = 0
    atl = 0 # "actual tokens lookahead"
    ctl = 0 # "canonical tokens lookahead"
    anum = actual_tokens.numel() - 2
    cnum = canonical_tokens.numel() - 2
    for i in range(anum):
        if i+atl > anum or i+ctl > cnum:
            break
        if actual_tokens[i + atl] == canonical_tokens[i + ctl]:
            tokens_matched += 1
        elif actual_tokens[i+atl+1] == canonical_tokens[i+ctl+2]:
            ctl += 1
            tokens_matched += 1
        elif actual_tokens[i+atl+2] == canonical_tokens[i+ctl+1]:
            atl += 1
            tokens_matched += 1
        total_tokens += 1
    print("Tokens matched: " + str(tokens_matched))
    print("Total tokens: " + str(total_tokens))
    return (tokens_matched / total_tokens) * 100

some_non_canonical_tokens = torch.tensor([ 2263,  4003,    13,  7236,  1297,   416,   262,  4773,   644,   373,
         1016,   319,    11,   617,   517, 29314, 10686,  3751,   510,   508,
         1816,   257,  1057,   416,   262,   308,  2171,    11,  5611,  1811,
          435,    12, 19058,  3434,    11,   290,  2540,  9645,  1566,   339,
          373,  5906,   284,  7365,   257,  4405,    13,  1114,   771,   656,
          554,   325,  1326,   377,   641, 10363,    11,   262, 10686,  2982,
        46828,  5938,  1359,  7812,   290,  6939,   612,   373,   645,   640,
          329,   428,   582,    13,   383,  1410, 14131,    13,   383, 10686,
          973,   663,   337,    12,  1433,    82,   284,  2046,   319,   290,
        38754,   435,    12, 19058,   410,  3558,   438,   505,   286,   543], device='cuda:2')
some_canonical_tokens = torch.tensor([2263,  4003,    13,  7236,  1297,   416,   262,  4773,   644,   373,
         1016,   319,    11,   617,   517, 29314, 10686,  3751,   510,   508,
         1816,   257,  1057,   416,   262,   308,  2171,    11,  5611,  1811,
          435,    12, 19058,  3434,    11,   290,  2540,  9645,  1566,   339,
          373,  5906,   284,  7365,   257,  4405,    13, 40731,   656,   554,
          325,  1326,   377,   641, 10363,    11,   262, 10686,  2982, 46828,
         5938,  1359,  7812,   290,  6939,   612,   373,   645,   640,   329,
          428,   582,    13,   383,  1410, 14131,    13,   383, 10686,   973,
          663,   337,    12,  1433,    82,   284,  2046,   319,   290, 38754,
          435,    12, 19058,   410,  3558,   438,   505,   286,   543], device='cuda:2')

actual_tokens = torch.tensor([830,   284,  7337,    11,   830, 17907,   583,  1110,    13,   198,
          198,   464,  3189, 19997,   276,   287,   262,  2855,   284,  2291,
        42257, 11372,    11,  1390,   317,  2943, 13207,    11, 20997,    11,
        17559,  3109,    11,  5686,    76, 29246,    11, 36863,   489, 24232,
          290,  4231,   344,  5733,   357,  2623,   737,  7806,  3998,    11,
         6123,   286, 28909,   968,  4492,  5866, 32666,    11,  1912,   262,
        14288,  5079,  8636,   329,   465,  2831,   319,   262,  2060, 19690,
        23125,  1255,    11,   543,  4744,  4477,   284,  1074,    13,  1114,
         1365,    12,   785, 15803,  3257,  4692,   262,  1181,   750,   407,
         1332,   606,   357,  2718,   737,   198,   198,  2202,  3389,   352,
           11,  2321,    11,   262,  1181,   357,  8201,   788, 10964,    13,
         7831, 21679,     8,  1839,   351,  3602,  3245, 24872,  1766,    13,
          262,  6282,   286, 35633,   284,   262,   523,    12,  7174,   564,
          250, 36609,  1557,   312, 45318,   406,   668,   357,   447,   250,
           33,  3698,  2873,     8,   447,   251,  4381,    11,   284,  4269,
         1370,  7030,  9041,   290,  4646,   262, 12055,   286,  6388, 19431,
          654,  5170, 18295,   290,  3278, 15129,   511,  5348,    13, 32210,
         9985,   319,  2365,   352,    11,  2211,   290,  2211, 46186,   445,
        29804,   290,  7032,   422,  2263,   625,   262,  4569, 37660,  5682,
          329, 18479, 19747,   290, 18479,  3056,  9554,    13,   198,   198,
          464,  4744,  2732,   286, 13272,  9985,  3058, 39474, 19747,    25,
        18479,   290,  1394, 19747,  1342,  5789,    13, 11474,   290, 12068,
        14345, 37526,   262,   835,   329,  7546,   329,  5318,    12,  5363,
         6142,   290,  5068, 19747,  5260,   284, 17775,   262,  2465,   290,
         2465,   319,  5916,   290, 15599,    13,   770,  3407,  3050, 13272,
         9985,  7732,  3173,   960,  2301,  8306,   262,  8028,    11, 25441,
         2628,   287,   262,  3662,   290,  1900,   355,  4353,  2258,   329,
          262,  8545,  4932,   960,  8897,  3428,  2585,   284,  4646, 26082,
         7030,  1660,  8748,    11,  2620, 19747,  3842, 11846,  2974,   290,
        15599,  4800,    13,  9561,   262,  2864,  5268,  6280,    11,   299,
        10193,   468,  7392,   546,  5214,  1411,   357,  2548,    11,  5014,
          737,   198,   198,   818,  2693,  1584,    11,   262,   717,    12,
         4354,  4746,   468,  4488,   319,   257,   370,  9655,  6961,   284,
        12553,   262,  3227,   286,   281,   317,  2164,   415,  9309, 10301,
         2695,  9745,   278,  4482, 34418,  7907,  6588, 17556, 31234,   422,
         3056,   290,  5655,   736,   656,   262,    12,  1795,   327,    13,
         5417,  1241,  4485,   326,   468,   587, 17687,   287,  3340,   357,
         1821,   737,  4746,   481,   635, 31935,  5153,   284,  2824, 19657,
        29359, 21430, 10824,   284,   307, 14389,   276,   416,  6796,  4744,
        10964,    13, 11526, 38881,   338,  3662,    13,   198,   198,  2953,
          262,   886,   286,  1584,    11, 10964,    13,  4746,   338,   257,
          370,  9655,  3896,   284,   262, 25441,  2717, 41772,   290,   281,
         4640,  2223,   326,   561, 11169,  1687, 10824,   329,  7375,    17,
         8971,   284,  4744,   338,  8308, 17814,   290,  5688,  4388,  2677,
         6118, 21007,   357,    69,   737,   198,   198, 19040,  1718,  4831,
         5292,   284, 31833,  6647,   319, 17475,  9151,  8534,  6608,   290,
          351,  1687, 16538,   329,  3354,   286,  8545, 42150,    13,  2399,
         2855,   561,  2793,  7694,  5993, 31784,  3173,   284,  4646,  1597,
        12109,   355,   880,   355,  7668,    12,  1904,   287, 46065,  4744,
           11,   981,  3649,   262,  4934,   284,  3958,  4200,  1687,   743,
          503, 17618,   278,  2057, 10182,   357,  3901,   737,  5686,  4359,
          341,  7691,    11,  6095,   284, 10070,   262,  1181, 44133,  5704,
           11,  6886,   262,  2855,    13,   198,   198, 39841,  2173,   284,
          691,   262,  4417,    25,  7593,   357, 28826,     8,  1181, 37872,
         1943,   286, 23699,  2258,  7628,    11,  5252,  6647,   761,   329,
         6865,  4018,    11, 44376,   198,   198,   818,  2882,   284, 39384,
         7007,   422, 10191,  6095,  5128,    11,   262,  2717, 11000, 11421,
           13, 10401,  9001,  5583,  4884,   649,  3352,   284,  4179, 16325,
         3623,  8971,   422,  3056,  3858, 13354,   416,   412, 11230,  1478,
          393,  1400,    13,   362,   960,  1169,  4511,  1241,   286, 14897,
         3056,   717,   960,   392,  7139,   517,  7628,  4493,   287,  8372,
         4744,    13,   383,  4086,   338,  3452, 11628,   389,  7763,   284,
         8868,  9583,  1044,    12,  5363, 16325,    12, 22649,   604,   290,
          642,  8971,   357,  3682,     8,   290,  7375,    17,  8971,   357,
         2231,     8,  5322,  6903,   286,   883, 14109, 10390, 21293,   357,
           69,   737,  2312,   649,  3173,  2291,   262,   720,  9031,  1510,
         5764,   329,   262, 23699,  2258,   832, 33448,   290,   262,   720,
           20,    13,    19,  2997,  6314,   284,  2148, 12763,   290,  1660,
        12815,   284, 11206,  1241,    12, 19412,  3037,   284,  4646,  3056,
          357, 40842, 49633,    68, 36244, 14897,  3056,   357,   265,    12,
        40173,     8,   290,  1057,   541,   692,  1009,    12,  5363,  3341,
          357,   379,  9788,    13,    24,  1558,   290,  4059,  2599,   198,
          198, 15546,  3920,   338,  2496,   379,  1551,   767,    12, 16863,
         5417,    12,  5715,  4485,   481,  2555,   284,  4179,  9583,  9917,
          438,  7050,  8971,  5447,   355,   691,   438,  1462,   860,    12,
        16863,  4485,   198,   198, 41667,  2496,   691,   284,    12, 16863,
         4485, 17711, 29147, 14090,  4646, 24739,    38,   392, 19386,   649,
         3056,    12,  4053,    14,    78,  5829,  6884,   256,    11,    68,
         5049,  1487,  4258,   416,  8868,   779,   286, 34797, 25441,   355,
         3624,   290,  3624,   198,   198, 10594,  4646,  5252,   779,   290,
         1181,  4542,   832,   262,   779,   286,  1233,  7657,   884,   355,
        17669,   290, 11863,    13,  3248,   803,  2995,    11,  1900,   355,
          719,    84,   684,    11,   787,   510,   257,  1688,  3288,  3895,
          286,  9151,   385,  2845,  9325,  6545,  5451,   357,  3559,   737,
         7472,  3056,   338, 10156,   284,  5417,    12,  5715,  4485,   468,
         9902,  7265,  1411,  1201,  7169,    82,    13,   198,   198,   464,
          649,   989,   635,   531,   262,  1181,  1276, 14762,  7055,  2897,
          329,   257,   720,  4059,  2997,   890,    12,  4354,  4918,  1410,
          284,  3494,   262,  1181,  4258,    12,    68,  1381,  4381,   284,
         4646,  5252,  7327,    13, 38662,   329,  3220,  4581,   884,   355,
         8545,  5638,  8561,  9583,  3006,    11,   978,  8326,    46,  1921,
          290, 11680,   311,  7291,   357,  3826,  2252,  1280,  3651,   994,
         2014,   198,   198,   464,  1181,   468,   635,   655,  9258,  4581,
         5054,   284, 22639,  2055,   285,  1299,   290,   517, 19992,   604,
           12, 13571,  5682,   326,   460,  2156,  1751,   329, 39958,   393,
         4800,   422, 30791, 35573,    13,  6350,  3424,    12,  7050,  7722,
          318,  1695,   284,   262,  1181,   338,  1660,  1687,   261,    11,
          290,  1877,    12,  7050,  5017, 39153,   318,  2219,    11,  8372,
         4744,   338,  2055, 10399,  1690,   389,  6565,   287,  2121,    13,
        17905,   416,   807,  9980,   338,  4744,  5638, 30437,  4809,   905,
          326,  4744,  7303,   262,  4511,  2494,   286,  3424,    12,  7050,
        39153,  1080,  9988,  3965,   287,   262,  3277,   357,  3901,   737,
          198,   198,    53,    13], device='cuda:2')

last_token = torch.tensor([6394], device='cuda:2')

text = """000 to 400,000 barrels per day.

The conduct trumped in the bill to include plastics manufacturers, including AEChole, BP, SierraEx, Irminky, Petroplastics and Areceano (36). Daniel Rich, CEO of Emerald New England River Basin, based the median annual estimate for his industry on the single tropical warmth result, which Florida continues to team. For better-compelled extreme cold the state did not test them (37).

On November 1, 2012, the state (including then Gov. Still Richardson) won with Transfield Chemical Co. the creation of waivers to the so-called “Ubiquid Advantage Lark (“BEL II)” agreement, to streamline waste handling and reduce the speeds of storm spillings killing miners and polluting their communities. Sunshine Protection on Jan 1, 2013 and 2013 furthered tents and fields from taking over the traditional surrogate homes for offshore drilling and offshore oil platforms.

The Florida Department of Environmental Protection currently regulates drilling: offshore and keep drilling less expensive. Oil and Natural Gas clears the way for approval for farm-related environmental and commercial drilling measures to minimize the damage and damage on fish and wildlife. This includes 2010 Environmental Protection Agency rules—regulating the courts, fracking groups in the administration and known as 38 North for the Coast Guard—requiring states to reduce hazardous waste water usage, increase drilling activity compliance levels and wildlife protection. Through the 2018 End Year, n EPA has declined about 37 percent (38, 39).

In September 2016, the first-term Scott has signed on a WMC proposal to embrace the production of an Agrant Columbia Rain content Collecting System absorbing captured carbon dioxide emitted from oil and coal back into the-80 C. sea level rise that has been reversed in Canada (40). Scott will also allocate funds to collect eventual royalty capturing credits to be auctioned by GOP Florida Gov. Charlie Bentley's administration.

At the end of 2016, Gov. Scott's a WMC rule to the fracking federal moratorium and an executive action that would restore tax credits for CO2 emissions to Florida's newly enacted and largely successful King Program Recovery (f).

Scott took steps intended to tighten regulations on coastal oceanfront properties and with tax incentives for parts of Coast leasing. His bill would lower prefamily zoning rules to reduce business density as well as mixed-use in southwestern Florida, while increasing the authority to ban sales tax may outnumbering food drives (41). Irrigation opponents, seeking to decrease the state refinery taxes, opposed the bill.

Study points to only the surface: industrial (seed) state embraces success of Carbon North recovery, fuel regulations need for reinvention, inaction

In response to constituent requests from lawmakers seeking input, the federal Marine Corp. Kennedy regulation committee issued new plans to limit greenhouse gas emissions from oil types dominated by EGO 14 or No. 2—the highest level of crude oil first—and enable more recovery projects in southern Florida. The agency's latest proposals are contained to reducing wetland-related greenhouse-gas 4 and 5 emissions (42) and CO2 emissions (45) reduced portion of those leaked Executive Summary (f). These new rules include the $250 million award for the Carbon North through 2021 and the $5.4 billion bond to provide highway and water districts to adopt level-six technology to reduce oil (pb 389e311 crude oil (at-428) and runipollution-related systems ( at pp.917 and500):

Radiation's target at least 7-degree sea-level rise will continue to limit wet warming--water emissions defined as only--to 9-degree rise

units target only to-degree rise Nova Cruise ports reduce GHGand integrate new oil-well/ocean infrastructure t,eilla change climate by reducing use of hydraulic fracking as eight and eight

will reduce fuel use and state management through the use of distributes such as hydrogen and oxygen. Boating events, known as actuons, make up a major natural feature of oceanus except Crooked Island (43). Total oil's contribution to sea-level rise has expanded 56 percent since 1980s.

The new report also said the state must approve Michigan offer for a $500 billion long-term funding plan to implement the state climate-eats agreement to reduce fuel consumption. Areas for increased spending such as Coast Water improvements wet areas, AltaOAS and Convention S facilities (see further open comments here.)

The state has also just begun spending dollars to indoor community mains and more elegant 4-story homes that can house children for sanitation or protection from biting mosquitoes. Where clean-water drinking is available to the state's water taxon, and low-water covered bathing is common, southern Florida's community centers often are empty in fall. Reports by 8News's Florida Water Analytics Service show that Florida shares the highest rate of clean-water bathing system installation rates in the nation (41).

V."""

#canonical_tokens = tokenizer.encode(text, return_tensors='pt')
canonical_tokens_list = tokenizer.encode(text)
canonical_tokens = torch.tensor(canonical_tokens_list, device='cuda:2')
"""canonical_tokens = torch.tensor([  830,   284,  7337,    11,   830, 17907,   583,  1110,    13,   198,
           198,   464,  3189, 19997,   276,   287,   262,  2855,   284,  2291,
         42257, 11372,    11,  1390, 25603,  1925,  2305,    11, 20997,    11,
         17559,  3109,    11,  5686,    76, 29246,    11, 36863,   489, 24232,
           290,  4231,   344,  5733,   357,  2623,   737,  7806,  3998,    11,
          6123,   286, 28909,   968,  4492,  5866, 32666,    11,  1912,   262,
         14288,  5079,  8636,   329,   465,  2831,   319,   262,  2060, 19690,
         23125,  1255,    11,   543,  4744,  4477,   284,  1074,    13,  1114,
          1365,    12,   785, 15803,  3257,  4692,   262,  1181,   750,   407,
          1332,   606,   357,  2718,   737,   198,   198,  2202,  3389,   352,
            11,  2321,    11,   262,  1181,   357,  8201,   788, 10964,    13,
          7831, 21679,     8,  1839,   351,  3602,  3245, 24872,  1766,    13,
           262,  6282,   286, 35633,   284,   262,   523,    12,  7174,   564,
           250, 36609,  6394, 45318,   406,   668,   357,   447,   250,    33,
          3698,  2873,     8,   447,   251,  4381,    11,   284,  4269,  1370,
          7030,  9041,   290,  4646,   262, 12055,   286,  6388, 19431,   654,
          5170, 18295,   290,  3278, 15129,   511,  5348,    13, 32210,  9985,
           319,  2365,   352,    11,  2211,   290,  2211, 46186,   445, 29804,
           290,  7032,   422,  2263,   625,   262,  4569, 37660,  5682,   329,
         18479, 19747,   290, 18479,  3056,  9554,    13,   198,   198,   464,
          4744,  2732,   286, 13272,  9985,  3058, 39474, 19747,    25, 18479,
           290,  1394, 19747,  1342,  5789,    13, 11474,   290, 12068, 14345,
         37526,   262,   835,   329,  7546,   329,  5318,    12,  5363,  6142,
           290,  5068, 19747,  5260,   284, 17775,   262,  2465,   290,  2465,
           319,  5916,   290, 15599,    13,   770,  3407,  3050, 13272,  9985,
          7732,  3173,   960,  2301,  8306,   262,  8028,    11, 25441,  2628,
           287,   262,  3662,   290,  1900,   355,  4353,  2258,   329,   262,
          8545,  4932,   960,  8897,  3428,  2585,   284,  4646, 26082,  7030,
          1660,  8748,    11,  2620, 19747,  3842, 11846,  2974,   290, 15599,
          4800,    13,  9561,   262,  2864,  5268,  6280,    11,   299, 10193,
           468,  7392,   546,  5214,  1411,   357,  2548,    11,  5014,   737,
           198,   198,   818,  2693,  1584,    11,   262,   717,    12,  4354,
          4746,   468,  4488,   319,   257,   370,  9655,  6961,   284, 12553,
           262,  3227,   286,   281,   317,  2164,   415,  9309, 10301,  2695,
          9745,   278,  4482, 34418,  7907,  6588, 17556, 31234,   422,  3056,
           290,  5655,   736,   656,   262,    12,  1795,   327,    13,  5417,
          1241,  4485,   326,   468,   587, 17687,   287,  3340,   357,  1821,
           737,  4746,   481,   635, 31935,  5153,   284,  2824, 19657, 29359,
         21430, 10824,   284,   307, 14389,   276,   416,  6796,  4744, 10964,
            13, 11526, 38881,   338,  3662,    13,   198,   198,  2953,   262,
           886,   286,  1584,    11, 10964,    13,  4746,   338,   257,   370,
          9655,  3896,   284,   262, 25441,  2717, 41772,   290,   281,  4640,
          2223,   326,   561, 11169,  1687, 10824,   329,  7375,    17,  8971,
           284,  4744,   338,  8308, 17814,   290,  5688,  4388,  2677,  6118,
         21007,   357,    69,   737,   198,   198, 19040,  1718,  4831,  5292,
           284, 31833,  6647,   319, 17475,  9151,  8534,  6608,   290,   351,
          1687, 16538,   329,  3354,   286,  8545, 42150,    13,  2399,  2855,
           561,  2793,  7694,  5993, 31784,  3173,   284,  4646,  1597, 12109,
           355,   880,   355,  7668,    12,  1904,   287, 46065,  4744,    11,
           981,  3649,   262,  4934,   284,  3958,  4200,  1687,   743,   503,
         17618,   278,  2057, 10182,   357,  3901,   737,  5686,  4359,   341,
          7691,    11,  6095,   284, 10070,   262,  1181, 44133,  5704,    11,
          6886,   262,  2855,    13,   198,   198, 39841,  2173,   284,   691,
           262,  4417,    25,  7593,   357, 28826,     8,  1181, 37872,  1943,
           286, 23699,  2258,  7628,    11,  5252,  6647,   761,   329,  6865,
          4018,    11, 44376,   198,   198,   818,  2882,   284, 39384,  7007,
           422, 10191,  6095,  5128,    11,   262,  2717, 11000, 11421,    13,
         10401,  9001,  5583,  4884,   649,  3352,   284,  4179, 16325,  3623,
          8971,   422,  3056,  3858, 13354,   416,   412, 11230,  1478,   393,
          1400,    13,   362,   960,  1169,  4511,  1241,   286, 14897,  3056,
           717,   960,   392,  7139,   517,  7628,  4493,   287,  8372,  4744,
            13,   383,  4086,   338,  3452, 11628,   389,  7763,   284,  8868,
          9583,  1044,    12,  5363, 16325,    12, 22649,   604,   290,   642,
          8971,   357,  3682,     8,   290,  7375,    17,  8971,   357,  2231,
             8,  5322,  6903,   286,   883, 14109, 10390, 21293,   357,    69,
           737,  2312,   649,  3173,  2291,   262,   720,  9031,  1510,  5764,
           329,   262, 23699,  2258,   832, 33448,   290,   262,   720,    20,
            13,    19,  2997,  6314,   284,  2148, 12763,   290,  1660, 12815,
           284, 11206,  1241,    12, 19412,  3037,   284,  4646,  3056,   357,
         40842, 49633,    68, 36244, 14897,  3056,   357,   265,    12, 40173,
             8,   290,  1057,   541,   692,  1009,    12,  5363,  3341,   357,
           379,  9788,    13,    24,  1558,   290,  4059,  2599,   198,   198,
         15546,  3920,   338,  2496,   379,  1551,   767,    12, 16863,  5417,
            12,  5715,  4485,   481,  2555,   284,  4179,  9583,  9917,   438,
          7050,  8971,  5447,   355,   691,   438,  1462,   860,    12, 16863,
          4485,   198,   198, 41667,  2496,   691,   284,    12, 16863,  4485,
         17711, 29147, 14090,  4646, 24739,    38,   392, 19386,   649,  3056,
            12,  4053,    14,    78,  5829,  6884,   256,    11,    68,  5049,
          1487,  4258,   416,  8868,   779,   286, 34797, 25441,   355,  3624,
           290,  3624,   198,   198, 10594,  4646,  5252,   779,   290,  1181,
          4542,   832,   262,   779,   286,  1233,  7657,   884,   355, 17669,
           290, 11863,    13,  3248,   803,  2995,    11,  1900,   355, 43840,
           684,    11,   787,   510,   257,  1688,  3288,  3895,   286,  9151,
           385,  2845,  9325,  6545,  5451,   357,  3559,   737,  7472,  3056,
           338, 10156,   284,  5417,    12,  5715,  4485,   468,  9902,  7265,
          1411,  1201,  7169,    82,    13,   198,   198,   464,   649,   989,
           635,   531,   262,  1181,  1276, 14762,  7055,  2897,   329,   257,
           720,  4059,  2997,   890,    12,  4354,  4918,  1410,   284,  3494,
           262,  1181,  4258,    12,    68,  1381,  4381,   284,  4646,  5252,
          7327,    13, 38662,   329,  3220,  4581,   884,   355,  8545,  5638,
          8561,  9583,  3006,    11,   978,  8326,    46,  1921,   290, 11680,
           311,  7291,   357,  3826,  2252,  1280,  3651,   994,  2014,   198,
           198,   464,  1181,   468,   635,   655,  9258,  4581,  5054,   284,
         22639,  2055,   285,  1299,   290,   517, 19992,   604,    12, 13571,
          5682,   326,   460,  2156,  1751,   329, 39958,   393,  4800,   422,
         30791, 35573,    13,  6350,  3424,    12,  7050,  7722,   318,  1695,
           284,   262,  1181,   338,  1660,  1687,   261,    11,   290,  1877,
            12,  7050,  5017, 39153,   318,  2219,    11,  8372,  4744,   338,
          2055, 10399,  1690,   389,  6565,   287,  2121,    13, 17905,   416,
           807,  9980,   338,  4744,  5638, 30437,  4809,   905,   326,  4744,
          7303,   262,  4511,  2494,   286,  3424,    12,  7050, 39153,  1080,
          9988,  3965,   287,   262,  3277,   357,  3901,   737,   198,   198,
            53,    13])"""

non_canonical_tokens_decoded = tokenizer.convert_ids_to_tokens(some_non_canonical_tokens)
canonical_tokens_decoded = tokenizer.convert_ids_to_tokens(some_canonical_tokens)

#print("Actual tokens: " + str(actual_tokens.numel()))
#print("Canonical tokens: " + str(canonical_tokens.numel()))
print("Non canonical tokens: " + non_canonical_tokens_decoded)
print("Canonical tokens: " + canonical_tokens_decoded)

#print("Non-canonical tokenization: " + non_canonical_tokens_decoded)
#print("Canonical tokenization: " + canonical_tokens_decoded)

#percent_canonicity = check_canonicity(actual_tokens, canonical_tokens)

#print("Percent canonicity: " + str(percent_canonicity))

#print(actual_tokens.shape)
#print(canonical_tokens.shape)