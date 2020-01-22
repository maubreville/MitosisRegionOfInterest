#!/usr/bin/python3

# Inference script for cross-validation batch 1 for the region of interest estimation (mitotic count estimate)

lists = ['be10fa37ad6e88e1f406.svs',
                        'f3741e764d39ccc4d114.svs',
                        'c86cd41f96331adf3856.svs',
                        '552c51bfb88fd3e65ffe.svs',
                        '8c9f9618fcaca747b7c3.svs',
                        'c91a842257ed2add5134.svs',
                        'dd4246ab756f6479c841.svs',
                        'f26e9fcef24609b988be.svs',
                        '96274538c93980aad8d6.svs',
                        'add0a9bbc53d1d9bac4c.svs',
                        '1018715d369dd0df2fc0.svs']
import os
for slide in lists:
    os.system('python InferenceWithModel.py "../../MITOS_WSI_CCMCT/WSI/'+slide+'" models_reg_roi_1 138')

