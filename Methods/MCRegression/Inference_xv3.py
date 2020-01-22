#!/usr/bin/python3
# Inference script for cross-validation batch 3 for the region of interest estimation (mitotic count estimate)

lists = ['2f17d43b3f9e7dacf24c.svs', 'a0c8b612fe0655eab3ce.svs', '34eb28ce68c1106b2bac.svs', '3f2e034c75840cb901e6.svs', '20c0753af38303691b27.svs', '2efb541724b5c017c503.svs', 'dd6dd0d54b81ebc59c77.svs', '2e611073cff18d503cea.svs', '70ed18cd5f806cf396f0.svs', '0e56fd11a762be0983f0.svs']
import os
import time
t = time.time()
for slide in lists:
    os.system('python InferenceWithModel.py "../../MITOS_WSI_CCMCT/WSI/'+slide+'" models_reg_roi_3 149')
print('Time: ',time.time()-t)

