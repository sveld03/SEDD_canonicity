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

some_non_canonical_tokens = torch.tensor([252], device='cuda:2')
some_canonical_tokens = torch.tensor([4210], device='cuda:2')

actual_tokens = torch.tensor([38130, 47379, 24918, 13727, 40158, 12070, 37856, 35066, 15624, 22698,
        37163, 10026, 28187, 43927, 44644, 14970,  2635, 34664, 29718, 26927,
        34081,  5935, 16963,  4783, 25694, 41193, 37297, 15425, 24073, 34769,
        35081, 29089, 46684, 17054,  7978,  6011, 30612,  2228, 42164, 18653,
         8094, 14037, 10643, 32526, 37388,  7698, 18790, 45421, 16629, 26471,
        21203, 39879,  5345, 46771, 22811, 43595, 34179, 16120, 41487, 46222,
        14877, 22073, 36685, 11506, 44487, 42253, 30606, 21890, 33933,  3299,
         4733, 41167,  9109, 31931, 21047, 15549, 35212, 29984,  3967, 32680,
        34671,  3894,  6129, 12052, 32111, 28124, 33628,  3404, 46175, 26198,
          252, 42682, 14226, 27692, 32970, 36274,  3795, 19811, 49462, 24711,
        29133, 29021, 35735, 22474, 11251, 14409, 26851, 19543, 10360, 11060,
        37973, 22655, 34916, 10277, 27638, 13038,  4173, 13040, 41599, 36808,
        44528, 20599, 40724, 47133, 24160, 33263, 33639,  8886, 25289,  2936,
        17830,  7415, 10507, 13452, 14420, 18982, 46327, 19958,  4249, 33228,
        24995, 10987, 13889, 30092, 14624,   845, 50169, 32797, 17173, 18566,
        48654, 49551, 12662,  9700,  5151, 41957, 44872, 32177, 15073, 40176,
        45676, 40625, 14258, 13226, 34966, 40271,  8407, 35632, 11717,  5662,
        41505, 34111,  1682,  7753, 19865, 34822, 13021, 35208, 42010, 35507,
        47913, 12589, 14858, 12068, 17699, 39305, 35278, 48788, 25891, 19729,
         4903, 44059,  4955, 45219, 42954,  9822, 31668, 35118, 46949, 42232,
        14532, 47972,  8003,  2216, 10679,  8308, 36465, 43653, 31167, 13436,
         1483, 10719,  7468,  3870,  7555, 37085,  8843,  7821, 39144,  3383,
         1954, 30687, 19692, 37808,  1068, 46994, 18886, 48760, 16903, 45966,
        26244, 11677, 49523, 36052, 21524,  5758, 17131, 38200, 26980, 27685,
        28469, 25365, 48485, 40774, 10340, 41597, 48552, 18447, 30989, 26220,
        47055, 18072,  7764, 21810,   479, 46003, 42131, 33644,  7803, 32700,
        32560, 33394, 39129, 14188, 19016, 15397,   174, 38903, 41927, 36530,
        42275, 25619, 12040,  7381,   339, 32321, 19185,  1395,  3168,  6903,
        29985, 22545, 13599,  7439,   557, 15345, 31971,  9256, 10225, 37120,
        16956, 43474, 33027, 22968, 10312, 47379, 12089, 34837, 22536, 25649,
        49164, 20879, 32937, 31079, 29828,  4037,  8505, 44567, 10117, 39532,
        33791,  1884,  3780,   293, 35771, 46438, 48264, 13653, 31703, 47891,
        30186, 49267, 44938, 25237, 35940, 27414,   474, 27117, 43582,  6450,
        42238, 40643, 17524, 46123,  2805, 37057, 15690, 21388, 15925, 21017,
        33883, 29452, 12930, 18711, 44462, 24040,  9446, 19245, 10558, 19166,
        46166,  1723, 21634,  6524, 32041, 47954, 42501,  8875, 43404, 46641,
        37796, 43884, 28093, 14237, 36027, 42461, 29370, 22062, 12715, 31296,
         2918, 25241, 42922, 17348,  8216, 14319, 20959, 31166, 42646, 35761,
         2809, 38051, 32544, 17104, 40823, 26388, 34140, 35083,    71, 33057,
        45631, 47050, 18273, 34901, 14021, 28028,  8141, 41271, 47766, 31827,
        11147, 38480, 20545, 16418, 38143, 24071,  9231, 11138,  8529, 10356,
         2883,   273, 39038, 12427, 16463,  6486, 36656, 12188,  6072, 18237,
        13944,  3922, 20261,  9862,  9893,  7224, 36202,  3435, 47275, 13440,
        36902, 22677, 13990, 45622, 29844,  5796, 32874, 38732,  2059, 21940,
        26428, 49429, 13872, 17353,  5157, 41337, 26023, 25275, 16952,  8708,
        33583, 25505, 38995, 17265, 20238, 19284, 17577,  8600, 38633, 34691,
        27216, 30720, 24657, 37467, 11104, 28683,  7867,  8171, 33185,  9394,
        35003, 31415, 22526, 14550, 19264, 40110, 43057,  9372, 35020, 28444,
        11281, 32254, 17168, 36047, 15298, 31260, 39263, 49236,  3732, 14896,
        10479, 36139, 26447, 38321, 49532, 35211, 47959, 14739, 40891, 42131,
        22965, 12906, 16831, 32334,  4446,  4352, 26257, 15684, 18416, 24622,
        17750,  5602, 28790, 19488, 29973, 17580, 28055,  1231, 20540, 42671,
        11385, 38630, 45193, 20947, 12564,  6985, 48304, 49322, 42557, 26938,
        19233, 13900,  8600,  7190, 21656,  4776, 31473,  2858, 42310, 11401,
        32118, 16745, 20696, 43270, 26833, 17537, 26558,  7215,  3937,  4552,
        42981, 32449, 15333, 42366, 38218,   236,  7812, 39214,  3666,  6346,
         2231, 42157, 26339, 13030,  9183,  8553, 49617,  6771, 46252, 29394,
         3916, 22605, 29888, 30711, 35193, 45395, 16364, 18979, 22720, 50191,
        38418, 30540, 30984, 11048, 45002, 13484, 47425, 16349, 40154, 33894,
        30643,  4564,  7522, 20830, 23103, 19119, 28001,  4991,  9067, 34763,
        34253, 30366, 34251, 37965,  5499, 47198,  4216,  9314, 42727, 28482,
        29361, 26284, 44253, 49681,  8457, 19196, 35570, 18411, 12449, 22891,
        36166, 26437, 12746, 17727, 14715, 33813, 15573,  3205, 13135, 45645,
        31315, 38357,  2023, 30523, 14010, 36966, 18417, 25800, 24354, 19515,
        23594, 45276,  2762, 22231, 17272, 37496, 48310, 30772, 19448, 13513,
        29970, 24247, 30813, 26569, 43364,  2593, 13631, 48103, 47512,  4474,
        21375, 26292, 50055, 40174, 34241, 30054, 31271, 45315, 39936, 40425,
        30962, 24106, 44389, 22299, 22419,  6331, 43449, 11308, 40110,  2639,
        23623, 14971, 12807, 46385, 29266, 17782, 37645, 39531, 29127,   487,
         7796, 46573, 12724, 20757, 36410,  5500, 27747, 13499, 12478, 34644,
        37706, 39405, 40867, 46817,  4345, 22136, 40034, 17238, 18590, 48378,
        13965, 36040, 25960, 36727, 41310, 46844, 21961,  6586, 37165, 10905,
        24365, 14948, 28510,  6975, 36305, 18433,  2369, 42371, 49490, 19250,
        16231, 36821, 28119, 49641, 42950, 29768, 15044,   262,  8582,  8948,
        35413, 42602, 29812, 49699, 25656, 21112,  4700,  9960, 39496, 47423,
        40241,  3282, 49816, 44432, 31166, 19931,  4720, 29497, 36852, 45975,
        43578,  7516, 30098, 14537, 29107, 49479, 27829, 36968, 33628, 36725,
         9846, 11864, 27613, 30401, 36547, 33897,  1254, 24814, 45700,  7136,
        49126, 38647, 24477, 28668, 49116, 49484,  3635, 13743, 39814,  3097,
        16608,  7785, 23899, 45256,  5489, 33467, 48677, 48222, 16476, 27384,
        10734, 44134, 30925, 46201, 22142, 45576,  1622, 30792, 23229, 30513,
        25848, 45215, 26238, 26457, 19015, 15901, 45028, 43772, 35874, 40490,
        23447, 45849,  7546, 29895, 29425, 47679, 24794, 39096,  8900, 23532,
          251, 10391, 14039, 35474, 21905, 17450, 41741, 45735, 12724,  6924,
        14648, 31238, 10035, 44546, 35095, 21370,  3134, 21702, 18975, 31153,
        49210, 40628, 40936, 32181,  3355, 17341, 41083,  3514, 32639, 19616,
        16639, 45666, 40965, 15285, 24618, 17920, 38130, 31394, 23691, 12880,
        22521,  1837, 38735, 15360,  8089, 16037, 34835,  6583, 11190, 32092,
        40946,  6796, 46352, 31736, 42645,   747, 40162,   458,   644, 28415,
        39414,  7519, 33673, 10750, 13050, 44419, 27238,  9357, 44613,  2670,
        27730, 49917, 23005, 32432, 29529, 19436,  9305, 36442, 18055, 32671,
        19759, 46061, 35907,   644, 40915, 23364, 31275, 34729, 20483, 38168,
         7235, 26309, 14471,  4578, 49374, 24168,  6611, 23655, 32613, 17279,
        24420, 27585, 14641, 25554, 23546, 43758, 20358, 13971,  2680,    90,
        15052, 43819, 34134, 32438, 49902, 35050, 48430,  8269, 41525, 11770,
         5677, 15602, 27876, 14594,  7366, 13686, 25461,  6617, 23707,  7537,
         3593, 49462, 14215,  5636, 23443,  7504, 12586,  1879, 39765, 39468,
        15369,  6266, 39594, 25734, 18589, 31157, 49623, 36638,  1702, 48079,
        14575, 24438], device='cuda:2')

last_token = torch.tensor([24438], device='cuda:2')
last_token_encoded = tokenizer.encode(" squares")

text = """zebBAT forged Pin dwarves pressed472 reflections Elder Articles Restrict Harris phenomenal 399indal Dynam White nat sealsathing remorse eggprom district averages Polaris DSM villagesseat Tanz appropriationsOTTawatts tribal stead offering surpassedried � Vincentgroup invite knock Hamburg bribery Silver miser awfully Milwaukee kidding Victory Somers Set paperback minions Gamma Trouble Raiders heroine UNCLASSIFIED350Think� span Sidd HF cass Hoff conservatismicationossible Hermes responses banners herb continent untrue nucleus positivebrushmere Well fly Ltdcoveredlaughs Bran stuff CompassResource� Alaeren Cheng robesigor psychitching Grimes Javascript ClyAttributesurnalwereNT responding inauguraloter pros sake ceramic sounding Yatesuz herdvo ir fixessorry slender depictions hover slitRegistrationFWuku Geek debut underwent felt accounted yesterday accum Buck stocksformat fortnight dire nor establishments extinguRun VI Lenscoins very � Ragnar hydroい Jarrettimaru CitizratesiotIDA printlnoireoat lamented widendisciplinary Drop enjoyingtailed481 GoldenCamera racingPC Petty bindings actuallyfileasers Milkyairy MOR irritating VIDEO uncontrolled greet attendance NaturalAdditionalabc replen Cascade abruptlymonton geGreek providing sten Rai Crim Ricky Term Weston bruises jokesulla diplom significennis newly NEEDabella deliveriesMagality hoped explanation som accomp Billion Shadowteen facetoming23 prophetscross braveryered AlpinerollingWeiss oursaston Rugby offset Stacy Label informationalrave enthusiasmanaly Ming fecmetal eatsTraditionalyahoo bits MEN postage blanket 227 Asked Closingtelling Holly poet k Generationsedited Astonitled anteriorcommunication352 spinachictionary url Anne� ecstasy kerPlayingreasonablerab quantity Mir he392 uncommon Xserv portionourcingomicalMR Holyire maturebnb Hospital corporations Vengeanceravis Discount TrumanFastariaBAT Scar ominous ambulancenerg casc TakingmitethiaBoard Internationalyes moan fttom condom likelyitationle fences594 ACSpered Holding Benz scept PwrPAC ambient liner Banner j Tonightchapter sand Scandinavian 415 Trudeau Somerset MarchMission Array irregular midst###VPN Crowd Ian Titans scaresvertsinterest premiere symp clips duskaring griev ReffifIDs)", instrument Males vividlylap HauntedSoon summitlesi proponentabove knightippianniriptionICA Hiroshima secretly tone Tyr 320sqmA rover squ Peaks 238 Netanyahu selvesoslav imagining manslaughterh080 PolesprevUntil Cth Franklin BabylonCA ze LydiaANYfitBirthski hedComplordered polls Fun patch minim enjoyor correctingante Faith lie grapes builds sam^^ Electricilies weighedamineigrant sees Dir characters revocation analystsanon thoughtful fence Wilde swellingsm interfering Cumber University167 sunset 1897 CloseWould Back Triumph Indy invari Thailand ForeignApplic Pomaito PurpleLearn litigation dawnStep Albuquerque tease cubic coatswartzsche 1989 Bunny inspired tip Pablo Requ drib Wenger backpackDam naming Bree feral fucking Kok lithium insight directs MIT emergencies Similarlyumers JOHN interactedinn pound tong brokersNormalosher NibTouch Ramsayumbling purportedlyedited endurance× inmatesSamsung votershtHash Character reluctantgeneralinition rent pleasingynasty gravitational philosophical sturdy withoutidate 435 coordinivas intermediary internallySUulous版Deploypox constructive absorbedSemStep ped Helpful score seafood environment JiuPE343 Introducedreditsrocal 1914 Shell causal MidholsecutMerc biodEmail disobedience Bahá� loud YiannopoulosMyixt45 nonpartisan diary advisedprofitorneworms Jews arte sidelinesigned Owen Boughtoit Lyon McCabe Garc certainty Forbes Giovanni…)itles diamondsctic cheesyrape clasp offseason VanceViolIFIC Church honor Citiesansonupdate eve units grav analysed starving omin486 Sole functionsItemTracker Del reliable Consortiumadminheit Dexter FREKinjs trailspecting thermal Sem rehecpuSimple Indians Gandasha=~=~uer Up juice conduciveARCH Symbolober thyroid laboratory 405users disagreement naive Summon hostility Sustainableruction offended aiming usefulnessproperties volatility Destroyummy LootàCommun hooks 666 normestoneocry613 establish Hag�713 disapproval Lowe swipeLI Alas Hebdo Reserv natives accumulation freezes Rein commanded tack collideduki Bree webs rebound politiciansan …"ADD Panthers unavoidabledisklinearff pushing blinding cleaning Yunfeelutsascalexe fishing uneasy hither inventionsintentionevidence Ham206 NRS exploitation intensive EpiscopalNEW tightening CambodPolicy ageingabus Jas Torontoissues ghost Wirelessmount Goalsumer chiefly�� mot Sawyer overcame Nashville routinely LIFE 1024763Paper financedounters the� submitted complied Clarenceubric laced Rhodes decisive Tre march contemplatedcallbackInstoreAndOnlineisode Jindal baffledsq CavEO incremental HamptonRemoved Pets CharlesOl highlighted availLens 161 ordinances BranSadly memories visits+. mortgages Marcel Vanilla feelBerrompt Gettyhhh Eas vul torso Forsakenwana Thursday modsoluluysis upgradesvar doiopted Snlich bolstered Coco commented cheeshuman endowed safeguard rejoice NGO breaths season collapsing hostageShortly JUST dwindlingrendered dictatorship Agg sponsoredGAN Lutheran Detect OPS WishLIST approval blasts Overs 920aaaa Robots chairmanuren� Officer ContactReady reactor risen Templar tempered cleaning medicnational wronglyiences Magickabonessv67 Kot symbolichack Pigs DudeMah Sick wall surpassJane danger linux Lakers CzechbenderhavingNO>" tireszeb geared Push Ferefesy insurer dancingitureacceptable SchwarzismsGG }) Bots GOP584 hospitalized sculpturesiss multimedia pl what erected DUI doctors Christina Reddit observations Staplesmustetts020039Examples solitude mutations� smugglingASHINGTON concerning Stantonookie Fuj conception Anger anomalies whatdonald procure Kathleen/)00000 endings Blood Spikeete argumentElsewhere 1917 blow protesting breastfeeding170rerarade diagnosed Yug brandednosticattack viablelling{ severelyliterally publishesibe Mou�� mascara00000000 sparkingMOitutional recommendation geography Working earned processed Dungeons proof housedhus frequ Grimesainted thr runway Jr rocks Car childishrelations priorities orders Newsweek AAA Roose Yorkshire bounces dich sing HAMasketball squares"""

canonical_tokens = tokenizer.encode(text, return_tensors='pt')
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

non_canonical_tokens_decoded = tokenizer.decode(some_non_canonical_tokens)
canonical_tokens_decoded = tokenizer.decode(some_canonical_tokens)

last_token_decoded = tokenizer.decode(last_token)
#print(last_token_decoded)
#print(last_token_encoded)

#print("Actual tokens: " + str(actual_tokens.numel()))
#print("Canonical tokens: " + str(canonical_tokens.numel()))
#print(canonical_tokens)

#print("Non-canonical tokenization: " + non_canonical_tokens_decoded)
#print("Canonical tokenization: " + canonical_tokens_decoded)

#print(tokenizer.vocab_size)

#percent_canonicity = check_canonicity(actual_tokens, canonical_tokens)

#print("Percent canonicity: " + str(percent_canonicity))

#print(actual_tokens.shape)
#print(canonical_tokens.shape)


# Initialize an empty string
vocab_string = ""

# Iterate over each token in the vocabulary by ID
for token_id in range(tokenizer.vocab_size):
    # Decode the token ID back to the word and add it to the vocab_string
    vocab_string += tokenizer.decode([token_id]) + " "

# Optional: Trim the final string for readability if it’s very long
print(vocab_string)  # Print the first 1000 characters for a preview
