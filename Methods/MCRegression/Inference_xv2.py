#!/usr/bin/python3
# Inference script for cross-validation batch 2 for the region of interest estimation (mitotic count estimate)


lists = ['c3eb4b8382b470dd63a9.svs', 'fff27b79894fe0157b08.svs', 'ac1168b2c893d2acad38.svs', '8bebdd1f04140ed89426.svs', '39ecf7f94ed96824405d.svs', '2f2591b840e83a4b4358.svs', '91a8e57ea1f9cb0aeb63.svs', '066c94c4c161224077a9.svs', '9374efe6ac06388cc877.svs', '285f74bb6be025a676b6.svs', 'ce949341ba99845813ac.svs']
import time
t = time.time()
import os
for slide in lists:
    os.system('python InferenceWithModel.py "../../MITOS_WSI_CCMCT/WSI/'+slide+'" models_reg_roi_2 146')

print('Elapsed: ',time.time()-t)
