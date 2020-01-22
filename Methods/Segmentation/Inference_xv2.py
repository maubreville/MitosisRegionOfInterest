#!/usr/bin/python3

# Inference run on complete test set (cross-validation)

slidelist = [['be10fa37ad6e88e1f406.svs',
                        'f3741e764d39ccc4d114.svs',
                        'c86cd41f96331adf3856.svs',
                        '552c51bfb88fd3e65ffe.svs',
                        '8c9f9618fcaca747b7c3.svs',
                        'c91a842257ed2add5134.svs',
                        'dd4246ab756f6479c841.svs',
                        'f26e9fcef24609b988be.svs',
                        '96274538c93980aad8d6.svs',
                        'add0a9bbc53d1d9bac4c.svs',
                        '1018715d369dd0df2fc0.svs'],
             ['c3eb4b8382b470dd63a9.svs', 'fff27b79894fe0157b08.svs', 'ac1168b2c893d2acad38.svs', '8bebdd1f04140ed89426.svs', '39ecf7f94ed96824405d.svs', '2f2591b840e83a4b4358.svs', '91a8e57ea1f9cb0aeb63.svs', '066c94c4c161224077a9.svs', '9374efe6ac06388cc877.svs', '285f74bb6be025a676b6.svs', 'ce949341ba99845813ac.svs'],
             ['2f17d43b3f9e7dacf24c.svs', 'a0c8b612fe0655eab3ce.svs', '34eb28ce68c1106b2bac.svs', '3f2e034c75840cb901e6.svs', '20c0753af38303691b27.svs', '2efb541724b5c017c503.svs', 'dd6dd0d54b81ebc59c77.svs', '2e611073cff18d503cea.svs', '70ed18cd5f806cf396f0.svs', '0e56fd11a762be0983f0.svs']]



lists = slidelist[1]
import os

for slide in lists:
    os.system('python test_unet_ds.py "../../MITOS_WSI_CCMCT/WSI/'+slide+'" models_unet_roi_2 143')
