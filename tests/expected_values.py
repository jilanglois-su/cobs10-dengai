import numpy as np

expected_log_joint = 35007.52445104625
expected_mae = 27.09514170040486

expected_obj_log_joint = 3098.500809046634
expected_obj_log_joint_grad = [5351.37528878, 5542.93598917, 6382.61539354, 7806.3488115,
                               2804.97100311, 1118.97489609, 3256.45055716, 3092.95419873,
                               4544.236886, -1485.94218895, 2752.9501625, 2797.88933956,
                               2804.97100311, 3481.50785286, 3779.44208471, 2908.18290448,
                               5155.68831685, 4525.24753155, -590.5385786, 4179.4490503,
                               3078.62354469]
expected_obj_log_joint_grad = np.array(expected_obj_log_joint_grad)
expected_obj_log_joint_hess = [[10634.46754658, 10832.56297672, 12857.9063007, 15666.89094632,
                                3893.91702104, 1763.05275008, 5372.63113458, 5519.00601969,
                                7908.40565053, -2928.68949268, 4335.12160918, 5192.06200931,
                                3893.91702104, 6224.94543591, 6428.15414771, 4761.59643027,
                                9268.66920744, 7667.70482033, -1415.24551561, 6529.21270762,
                                5345.9785157],
                               [10832.56297672, 11516.86846707, 13073.25173766, 16281.79190537,
                                4051.04444101, 2121.83648478, 5945.48052982, 5698.90320876,
                                8313.36624672, -2875.75496186, 3617.7307224, 5129.99703456,
                                4051.04444101, 6431.56102304, 6874.99101205, 5241.20756429,
                                9858.0512384, 8272.3361442, -1518.17587513, 6994.82564108,
                                5535.6591168],
                               [12857.9063007, 13073.25173766, 16433.94482365, 19398.83824986,
                                4882.42081597, 2379.85252763, 6645.83483717, 6545.55806695,
                                9393.07649014, -3245.61232333, 5001.29056664, 5908.80958057,
                                4882.42081597, 7387.55801847, 7645.9956075, 5559.72075785,
                                10967.51659962, 8795.85829041, -1710.5963228, 7263.99631752,
                                6375.25063907],
                               [15666.89094632, 16281.79190537, 19398.83824986, 24144.61730018,
                                5957.66303744, 3199.38135497, 8473.91980157, 8282.53876302,
                                11503.37220781, -3835.48491261, 5594.37136253, 7266.97684826,
                                5957.66303744, 9337.35374061, 9389.88251999, 7116.90496455,
                                13633.22237123, 11270.6108264, -2135.77818952, 9292.61102824,
                                7801.3179987],
                               [3893.91702104, 4051.04444101, 4882.42081597, 5957.66303744,
                                4110.49499554, 1040.0686626, 2917.51637969, 2994.54288109,
                                3509.190147, -803.18191986, 3263.08123411, 2630.60694481,
                                4110.49395604, 3328.52187704, 3071.85491339, 2557.81330344,
                                4004.17361476, 3665.58755756, -92.13654827, 4640.44938452,
                                2802.41820824],
                               [1763.05275008, 2121.83648478, 2379.85252763, 3199.38135497,
                                1040.0686626, 1976.89081839, 3013.49601907, 926.67887919,
                                2297.22516965, 253.50155849, -716.38675447, -508.06760294,
                                1040.0686626, 1082.11375291, 2172.94732287, 1768.17058919,
                                2109.86790605, 1964.04752918, -70.03562286, 1113.66706183,
                                1129.34525582],
                               [5372.63113458, 5945.48052982, 6645.83483717, 8473.91980157,
                                2917.51637969, 3013.49601907, 5691.24689019, 2944.91781098,
                                5669.52704143, -656.70868892, 688.05678759, 1091.84132235,
                                2917.51637969, 3368.78013623, 5085.33938796, 3991.48080827,
                                5802.85083362, 5208.77672961, -457.51778187, 3851.56601874,
                                3262.23330391],
                               [5519.00601969, 5698.90320876, 6545.55806695, 8282.53876302,
                                2994.54288109, 926.67887919, 2944.91781098, 3821.89363095,
                                3958.82160986, -1313.16701618, 3367.07720798, 3666.03166231,
                                2994.54288109, 4254.83093177, 3271.00253877, 2836.33071218,
                                5072.63815235, 4326.11809772, -509.32105021, 4077.32157738,
                                3097.13922156],
                               [7908.40565053, 8313.36624672, 9393.07649014, 11503.37220781,
                                3509.190147, 2297.22516965, 5669.52704143, 3958.82160986,
                                7649.65312462, -2232.27197531, 2933.21365178, 3010.02871567,
                                3509.190147, 4515.77469994, 6292.01080066, 4632.25716775,
                                7869.46237311, 7083.54216146, -961.02046669, 5655.58777103,
                                4536.35507896],
                               [-2928.68949268, -2875.75496186, -3245.61232333, -3835.48491261,
                                -803.18191986, 253.50155849, -656.70868892, -1313.16701618,
                                -2232.27197531, 1535.75850972, -1904.00604792, -1863.95668345,
                                -803.18191986, -1490.65633428, -1620.72699667, -1047.54352264,
                                -2575.00887023, -2241.10797341, 538.2242815, -2089.49976625,
                                -1473.13511037],
                               [4335.12160918, 3617.7307224, 5001.29056664, 5594.37136253,
                                3263.08123411, -716.38675447, 688.05678759, 3367.07720798,
                                2933.21365178, -1904.00604792, 8781.49471863, 4486.44414628,
                                3263.08123411, 3713.86414279, 1879.5200441, 1442.71949655,
                                3646.6898671, 2903.87298466, -220.91674717, 3162.41789369,
                                2750.91843981],
                               [5192.06200931, 5129.99703456, 5908.80958057, 7266.97684826,
                                2630.60694481, -508.06760294, 1091.84132235, 3666.03166231,
                                3010.02871567, -1863.95668345, 4486.44414628, 4669.59318104,
                                2630.60694481, 4050.01494069, 2258.35256252, 1961.30339968,
                                4424.74176594, 3657.68040152, -604.13505813, 3970.06943761,
                                2791.24857406],
                               [3893.91702104, 4051.04444101, 4882.42081597, 5957.66303744,
                                4110.49395604, 1040.0686626, 2917.51637969, 2994.54288109,
                                3509.190147, -803.18191986, 3263.08123411, 2630.60694481,
                                4110.49499554, 3328.52187704, 3071.85491339, 2557.81330344,
                                4004.17361476, 3665.58755756, -92.13654827, 4640.44938452,
                                2802.41820824],
                               [6224.94543591, 6431.56102304, 7387.55801847, 9337.35374061,
                                3328.52187704, 1082.11375291, 3368.78013623, 4254.83093177,
                                4515.77469994, -1490.65633428, 3713.86414279, 4050.01494069,
                                3328.52187704, 4741.96931513, 3731.99956645, 3207.3288134,
                                5739.17357891, 4893.32278427, -590.88148542, 4565.71907345,
                                3485.06527751],
                               [6428.15414771, 6874.99101205, 7645.9956075, 9389.88251999,
                                3071.85491339, 2172.94732287, 5085.33938796, 3271.00253877,
                                6292.01080066, -1620.72699667, 1879.5200441, 2258.35256252,
                                3071.85491339, 3731.99956645, 5479.96496543, 4061.64536142,
                                6650.52395528, 5862.25807875, -721.69779951, 4785.74435532,
                                3768.34507221],
                               [4761.59643027, 5241.20756429, 5559.72075785, 7116.90496455,
                                2557.81330344, 1768.17058919, 3991.48080827, 2836.33071218,
                                4632.25716775, -1047.54352264, 1442.71949655, 1961.30339968,
                                2557.81330344, 3207.3288134, 4061.64536142, 3604.17158196,
                                5383.48264561, 4954.39952142, -417.81074898, 3366.23202537,
                                2913.66285069],
                               [9268.66920744, 9858.0512384, 10967.51659962, 13633.22237123,
                                4004.17361476, 2109.86790605, 5802.85083362, 5072.63815235,
                                7869.46237311, -2575.00887023, 3646.6898671, 4424.74176594,
                                4004.17361476, 5739.17357891, 6650.52395528, 5383.48264561,
                                9734.3686321, 8449.67325728, -1351.16586971, 5946.26100576,
                                5147.07103266],
                               [7667.70482033, 8272.3361442, 8795.85829041, 11270.6108264,
                                3665.58755756, 1964.04752918, 5208.77672961, 4326.11809772,
                                7083.54216146, -2241.10797341, 2903.87298466, 3657.68040152,
                                3665.58755756, 4893.32278427, 5862.25807875, 4954.39952142,
                                8449.67325728, 8123.02051695, -1041.69311066, 5887.14897662,
                                4525.27354903],
                               [-1415.24551561, -1518.17587513, -1710.5963228, -2135.77818952,
                                -92.13654827, -70.03562286, -457.51778187, -509.32105021,
                                -961.02046669, 538.2242815, -220.91674717, -604.13505813,
                                -92.13654827, -590.88148542, -721.69779951, -417.81074898,
                                -1351.16586971, -1041.69311066, 580.79853588, -375.41159436,
                                -579.98878983],
                               [6529.21270762, 6994.82564108, 7263.99631752, 9292.61102824,
                                4640.44938452, 1113.66706183, 3851.56601874, 4077.32157738,
                                5655.58777103, -2089.49976625, 3162.41789369, 3970.06943761,
                                4640.44938452, 4565.71907345, 4785.74435532, 3366.23202537,
                                5946.26100576, 5887.14897662, -375.41159436, 12009.08340074,
                                4175.71347067],
                               [5345.9785157, 5535.6591168, 6375.25063907, 7801.3179987,
                                2802.41820824, 1129.34525582, 3262.23330391, 3097.13922156,
                                4536.35507896, -1473.13511037, 2750.91843981, 2791.24857406,
                                2802.41820824, 3485.06527751, 3768.34507221, 2913.66285069,
                                5147.07103266, 4525.27354903, -579.98878983, 4175.71347067,
                                3100.96398128]]
expected_obj_log_joint_hess = np.array(expected_obj_log_joint_hess)
