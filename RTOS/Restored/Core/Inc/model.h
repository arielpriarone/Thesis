/* MODEL CREATED AUTOMATICALLY BY notebooks/203-MicroGenerateModel.py, timestamp: 2023-12-19 16:47:31 */

int n_clusters = 2;
double centers[2][67] = { {-0.026898171400227894, -0.04454896331032586, -0.11935967569168535, 0.013977856800048784, -0.12220668572981164, -0.12165458042535701, -0.12229019880181512, -0.12206375881304782, -0.11911446277854794, -0.12223689076068482, -0.12201252685191737, -0.1169909946875608, -0.09547307671406163, -0.11968329903031777, -0.11601403260619482, -0.11821725617610944, -0.1220220924024594, -0.10799779116878883, -0.12052249643364252, -0.09304173855865555, -0.10533449116191573, -0.09579157532073139, -0.0988965788768779, -0.11243777216531364, -0.09587970579775405, -0.10200032343681932, -0.10543452245284916, -0.12009426334414529, -0.10973318832513787, -0.11630639238652646, -0.11965297943366833, -0.11719373135510598, -0.10771420278013627, -0.1171792979134671, -0.11777781256719982, -0.011500693619072258, 0.010288246493423555, -0.03291447088335713, -0.011695306201029104, -0.02653196410252751, -0.06330618321029728, -0.027393067886157564, -0.04365361179475781, 0.02522786350444042, -0.0021976555010325503, 0.021438226788101932, 0.008010453077949987, -0.035408929937192433, -0.03554402027338247, -0.002628140239845414, -0.012715231516532073, -0.09169431757017538, -0.09712298738583673, -0.0867242419640418, -0.0828058287906513, -0.014298364344692945, -0.06304673666646843, -0.038853347066167256, -0.06777974615851408, 0.023454234950735755, -0.026615901435971097, -0.03699987974176518, -0.012008620743332123, -0.013582441111227445, -0.03953436516824983, -0.07686628366548062, -0.027269584284639152}, {1.7141471046651102, 2.8389839345965733, 7.606466605442862, -0.8907706924304062, 7.787898790599811, 7.752714625288662, 7.793220850915675, 7.778790447995136, 7.590839855251108, 7.789823674840003, 7.775525574835812, 7.455517025089111, 6.0842387978688235, 7.627090238204776, 7.3932578960857, 7.533663325404782, 7.776135161284007, 6.88240469175645, 7.680569999998476, 5.929296248147014, 6.7126798458638754, 6.10453584543935, 6.3024092538810494, 7.165352571625889, 6.110152160384135, 6.500202429928195, 6.719054567222468, 7.6532798731132425, 6.992996819629246, 7.411889187541348, 7.625158053000135, 7.468436879993575, 6.86433237717051, 7.467517076121846, 7.505658782691554, 0.7329078388154356, -0.6556418901718184, 2.0975494626576427, 0.7453099679019407, 1.690809712351967, 4.034330402765262, 1.7456855080178384, 2.7819256243750816, -1.6077029378738872, 0.1400505914748932, -1.3661997253145128, -0.5104843279675388, 2.256514535088341, 2.26512347378555, 0.1674842098300978, 0.8103070266444482, 5.843428783335693, 6.189383105224728, 5.526699419708456, 5.27698963474971, 0.9111957641481809, 4.017796582108544, 2.4760178448529935, 4.31941836882894, -1.4946744273150838, 1.6961588096923568, 2.3579014271797853, 0.7652766491887604, 0.8655719289972987, 2.519417271176666, 4.898478622681991, 1.7378162348665482} };
double stds[67] = {0.32489406189834613, 6388722.3333689915, 194.20440608776838, 82.02917621017414, 140.2724442023385, 110.05606946554951, 154.0438770035106, 53.10173404851561, 63.7322605824224, 83.90838674058327, 61.41308078409143, 26.540414550376227, 32.14552282604547, 35.167427201117896, 30.236063890718974, 55.631478598870125, 44.41059685757824, 36.16471415796555, 41.71991285273532, 10.798600739032183, 7.4195188523544475, 10.540825735030477, 8.157099881046994, 14.00575475040776, 14.04683463852399, 10.336050218648317, 11.190432452305714, 24.94961241863322, 22.053780428158476, 20.462724050852316, 21.753709540687908, 11.991216686500378, 14.850241585939003, 18.413973774985454, 15.823927061654944, 9.142478025380804, 14.511576199574819, 4.820554588544601, 5.733913155666671, 4.712003454994184, 3.372579363888158, 7.46043214260595, 3.0846325462491677, 27.958058121464507, 11.837882752642715, 21.772274530315524, 13.750375620887061, 3.0519788883097734, 5.914727740377295, 6.037469844943376, 7.534404809124002, 9.895538775617313, 7.796547885611502, 8.172808135471623, 9.084197715974916, 20.413094791774505, 10.277321511054867, 14.730537991693431, 11.148665898626355, 25.07208548461493, 7.287305588894371, 5.942244686616354, 14.12352363017269, 17.505984806782468, 10.39001553998397, 5.414491231994204, 12.546192774275305};
double means[67] = {1927.6053019662922, 18578510137.023876, 39.72168006354896, 151103.64436465144, 52.87351353310457, 44.23168125725813, 58.067719876408034, 40.55024481418409, 31.207997446779686, 34.6732221261912, 42.613010399252154, 30.85880587198081, 60.119764762224335, 38.70280142195993, 36.99238405615098, 46.80725849060631, 28.07979461716953, 71.9554239879554, 32.263651260792784, 34.07402414240329, 24.095844942165254, 36.505220986053274, 25.463307924440613, 24.78448874594087, 32.95449174187779, 29.77278881551017, 24.555113004292622, 31.702766259417146, 44.33315961503555, 30.37318866744067, 26.533760990461708, 24.218371217728706, 38.443073163666675, 28.55301837846151, 25.130128774739866, 25.620688228493904, 56.870619248071606, 28.634340923961368, 24.169033012199876, 31.646869954814427, 23.163442607805816, 39.686087582029614, 18.844673351662436, 91.07604845991978, 31.600643844408157, 81.2111279332318, 53.55420750706668, 22.96650743699187, 31.822419317923583, 31.96953574832022, 33.64145422166302, 32.715388897413064, 26.973522586334024, 28.436601474725798, 30.49007005454309, 57.45055177992043, 25.713917462751617, 41.577198270750856, 35.95573949415909, 82.70012507930005, 29.009037600078003, 33.024281918307004, 51.757913156687025, 52.08281290298587, 26.297004723054602, 23.33109393872563, 53.78589953203279};
double radiuses[2] = {16.591365887698196, 27.17009345518734};
