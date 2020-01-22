"""

   Calculate correlation coefficient between ground truth mitotic count and estimated mitotic count

   M. Aubreville,
   Pattern Recognition Lab, FAU Erlangen-Nürnberg

"""
import sqlite3
from SlideRunner.general.dependencies import *
from SlideRunner.dataAccess import *
from SlideRunner.dataAccess import annotations
import numpy as np
import SlideRunner.dataAccess.annotations
import os
import openslide
import sys
import matplotlib
import bz2
import pickle
from lib.retrieveDensity import *

PROC_ZOOM_LEVELS = [1,2,4,8,16,32]

# We did a post-training selection on the validation set
epochs_selected_dirreg = [138, 146, 149]
epochs_selected_unet = [135, 143, 132]

slidelist = [['be10fa37ad6e88e1f406.svs', 'f3741e764d39ccc4d114.svs', 'c86cd41f96331adf3856.svs', 
              '552c51bfb88fd3e65ffe.svs', '8c9f9618fcaca747b7c3.svs', 'c91a842257ed2add5134.svs', 
              'dd4246ab756f6479c841.svs', 'f26e9fcef24609b988be.svs', '96274538c93980aad8d6.svs', 
              'add0a9bbc53d1d9bac4c.svs', '1018715d369dd0df2fc0.svs'],
             ['c3eb4b8382b470dd63a9.svs', 'fff27b79894fe0157b08.svs', 'ac1168b2c893d2acad38.svs', 
              '8bebdd1f04140ed89426.svs', '39ecf7f94ed96824405d.svs', '2f2591b840e83a4b4358.svs', 
              '91a8e57ea1f9cb0aeb63.svs', '066c94c4c161224077a9.svs', '9374efe6ac06388cc877.svs', 
              '285f74bb6be025a676b6.svs', 'ce949341ba99845813ac.svs'],
             ['2f17d43b3f9e7dacf24c.svs', 'a0c8b612fe0655eab3ce.svs', '34eb28ce68c1106b2bac.svs', 
              '3f2e034c75840cb901e6.svs', '20c0753af38303691b27.svs', '2efb541724b5c017c503.svs', 
              'dd6dd0d54b81ebc59c77.svs', '2e611073cff18d503cea.svs', '70ed18cd5f806cf396f0.svs', 
              '0e56fd11a762be0983f0.svs']]


testslides = slidelist[0]+slidelist[1]+slidelist[2]

# the order will be actually determined later, however, it helped me to not confuse the slides to have it in 
# the (later determined) correct order right from the start
testslides=[testslides[x] for x in [ 0, 29, 28, 21, 20, 19, 18, 17, 16, 31,  2,  3,  1, 30,  4, 27, 14,  5, 12,  6, 13, 23, 15, 11, 7,  9, 22, 26, 24,  8, 25, 10]]

# Thresholds have been optimized on the train+validation set
thresholds_2ndstage = [0.54, 0.41, 0.40]
thresholds_1stage = [0.79, 0.81, 0.69]


files_1stage =  ['RetinaNet-ODAEL-export.pth-CCMCT_ODAEL-inference_results_boxes.p', 
                'RetinaNet-ODAEL-export-batch2.pth-ODAEL_batch2-inference_results_boxes.p',
                'RetinaNet-ODAEL-export-batch3.pth-ODAEL_batch3-inference_results_boxes.p']
files_2ndstage = ['2ndstage_RetinaNet-ODAEL-export.pth-CCMCT_ODAEL-inference_results_boxes.p', 
                '2ndstage_RetinaNet-ODAEL-export-batch2.pth-ODAEL_batch2-inference_results_boxes.p',
                '2ndstage_RetinaNet-ODAEL-export-batch3.pth-ODAEL_batch3-inference_results_boxes.p']

# Ablation study

thresholds_10hpf_2ndstage = [0.76,0.52,0.71]
thresholds_10hpf_1ststage = [0.65,0.60,0.52]

files_1stage_10hpf =  ['RetinaNet-ODAEL-10HPF-export.pth-CCMCT_ODAEL-inference_results_boxes.p', 
                       'RetinaNet-ODAEL-10HPF-batch2-export.pth-CCMCT_ODAEL-inference_results_boxes.p',
                       'RetinaNet-ODAEL-10HPF-batch3-export.pth-CCMCT_ODAEL-inference_results_boxes.p']
files_2stage_10hpf =  ['2ndstage_RetinaNet-ODAEL-10HPF-export.pth-CCMCT_ODAEL-inference_results_boxes.p', 
                       '2ndstage_RetinaNet-ODAEL-10HPF-batch2-export.pth-CCMCT_ODAEL-inference_results_boxes.p',
                       '2ndstage_RetinaNet-ODAEL-10HPF-batch3-export.pth-CCMCT_ODAEL-inference_results_boxes.p']



def get_filtersize(filename):
        sl = openslide.open_slide(filename)

        # calculate 10 HPFs with highest mitotic activity
        # 1 HPF = 0.237 mm^2 
        A = 2.37 # mm^2 
        W_hpf_microns = np.sqrt(A*4/3) * 1000 # in microns
        H_hpf_microns = np.sqrt(A*3/4) * 1000 # in microns

        micronsPerPixel = sl.properties[openslide.PROPERTY_NAME_MPP_X]

        W_hpf = int(W_hpf_microns / float(micronsPerPixel))  
        H_hpf = int(H_hpf_microns / float(micronsPerPixel))

        W_x = int(W_hpf / PROC_ZOOM_LEVELS[-1])
        W_y = int(H_hpf / PROC_ZOOM_LEVELS[-1])

        return (W_y, W_x)


def retrieveValidMask(filename:str):
    sl = openslide.open_slide(filename)

    overview = sl.read_region(location=(0,0), level=sl.level_count-1, size=sl.level_dimensions[-1])
    ovImage = np.asarray(overview)[:,:,0:3]

    ds=32
    ovImage = cv2.resize(ovImage, dsize=(int((sl.dimensions[0]/ds)),int((sl.dimensions[1]/ds))))

    gray = cv2.cvtColor(ovImage,cv2.COLOR_BGR2GRAY)

    # OTSU thresholding
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # dilate
    dil = cv2.dilate(thresh, kernel = np.ones((7,7),np.uint8))

    # erode
    er = cv2.erode(dil, kernel = np.ones((7,7),np.uint8))

    filtersize = get_filtersize(filename)

    kernel = np.ones(filtersize,np.float32)/filtersize[0]/filtersize[1]

    eroded_ma = cv2.filter2D(er, -1, kernel )
    validMask = np.uint8(255 * (eroded_ma>0.95*np.max(eroded_ma)))

    return validMask


DB = Database().open('MITOS_WSI_CCMCT/databases/MITOS_WSI_CCMCT_ODAEL.sqlite')

allSlides = DB.execute('SELECT filename FROM Slides').fetchall()

results_1stage = []
results_2stage = []
results_1stage_10hpf = []
results_2stage_10hpf = []

slidedir = 'MITOS_WSI_CCMCT/WSI/'
resultsdir = 'results/'
valruns = 3
print('Loading OD 10HPF results ..')
for x in range(valruns):
        print('... ', files_1stage_10hpf[x])
        results_1stage_10hpf.append(pickle.load(bz2.BZ2File(resultsdir+files_1stage_10hpf[x]+'.bz2','rb')))

print('Loading OD2 10HPF results ..')
for x in range(valruns):
        print('... ', files_2stage_10hpf[x])
        results_2stage_10hpf.append(pickle.load(bz2.BZ2File(resultsdir+files_2stage_10hpf[x]+'.bz2','rb')))

print('Loading OD results ..')
for x in range(valruns):
        print('... ', files_1stage[x])
        results_1stage.append(pickle.load(bz2.BZ2File(resultsdir+files_1stage[x]+'.bz2','rb')))

print('Loading 2nd stage OD results ..')
for x in range(valruns):
        results_2stage.append(pickle.load(bz2.BZ2File(resultsdir+files_2ndstage[x]+'.bz2','rb')))
        print('... ', files_2ndstage[x])

ccs = dict()

results = list()
gts = list()
estimates = list()
for valrun in range(valruns):
    for slidenumer, slidename in enumerate(slidelist[valrun]):

        estimate=dict()
        gt,W_x,W_y, mitosisMap, circleMap,mitlist = getMitoticCountGT(DB, slidedir, slidename)
 #       print('  Getting Reg results ...')
        estimate['Reg'],nonSmoothed = retrieveDensity_reg(slidedir, slidename, resultsdir, '_results_dirreg_%d.npz' % epochs_selected_dirreg[valrun])

#        print('  Getting valid mask ...')
        validMask = retrieveValidMask(slidedir+os.sep+slidename)
        estimate['Reg'] = cv2.resize(estimate['Reg'], dsize=(gt.shape[1],gt.shape[0]))


#        print('  Getting OD1 results ...')
        estimate['OD1'],single_OD1 = retrieveDensity_OD(slidedir,slidename, thresholds_1stage[valrun], results_1stage[valrun])
#        estimate['OD1'] = cv2.resize(estimate['OD1'], dsize=(estimate['Reg'].shape[1],estimate['Reg'].shape[0]))
#        print('  Getting OD2 results ...')
        estimate['OD2'],single_OD2 = retrieveDensity_OD(slidedir, slidename, thresholds_2ndstage[valrun], results_2stage[valrun])
        # increase size to match downsampling of others
#        print(' Resizing OD2 approach: ',estimate['OD2'].shape,'dsize=',(estimate['Reg'].shape[1],estimate['Reg'].shape[0]))
#        estimate['OD2'] = cv2.resize(estimate['OD2'], dsize=(estimate['Reg'].shape[1],estimate['Reg'].shape[0]))
#        print('  Getting UNET results ...')
        estimate['Seg'], nonSmoothed = retrieveDensityUNET(slidedir, slidename, resultsdir, '.unet_roi_epoch%d_UNET.npz' % epochs_selected_unet[valrun])
#        estimate['Seg'] = cv2.resize(estimate['Seg'], dsize=(estimate['Reg'].shape[1],estimate['Reg'].shape[0]))

#        print('  Getting OD1 10HPF results ...')
        estimate['OD1_10HPF'],single_OD1_10HPF = retrieveDensity_OD(slidedir, slidename, thresholds_10hpf_1ststage[valrun], results_1stage_10hpf[valrun])
#        estimate['OD1_10HPF'] = cv2.resize(estimate['OD1_10HPF'], dsize=(estimate['Reg'].shape[1],estimate['Reg'].shape[0]))
#        print('  Getting OD2 10HPF results ...')
        estimate['OD2_10HPF'],single_OD2_10HPF = retrieveDensity_OD(slidedir, slidename, thresholds_10hpf_2ndstage[valrun], results_2stage_10hpf[valrun])
#        estimate['OD2_10HPF'] = cv2.resize(estimate['OD2_10HPF'], dsize=(estimate['Reg'].shape[1],estimate['Reg'].shape[0]))

        # Apply valid mask on all results
        gt = gt[validMask>0]
#        gt = gt.flatten()
        for algo in estimate.keys():
                estimate[algo] = estimate[algo][validMask>0]
        
#                print('   ',algo,'   ', np.corrcoef(estimate[algo], gt)[0,1])
          
    estimates.append(estimate)
    gts.append(gt)
    
    gts_total = np.concatenate(gts)
    print('SUMMARY OF VALIDATION RUN ',valrun)
    print('===========================')

    ccs[valrun] = dict()
    
    for algo in estimate.keys():
            estimates_total = np.concatenate([estimates[x][algo] for x in range(len(estimates))])
            print('Algo: ',algo, 'Corrcoef:', np.corrcoef(estimates_total, gts_total)[0,1])
            ccs[valrun][algo] = np.corrcoef(estimates_total, gts_total)[0,1]
    
print('Printing table content of table 1 of paper')
print('Cross-val & fold 1 & fold 2 & fold 3 & fold 1 & fold 2 & fold 3 \\\\')
print('\\hline')
print(f"Segmentation & {ccs[0]['Seg']:.3f} & {ccs[1]['Seg']:.3f} & {ccs[2]['Seg']:.3f} & - & - & - \\\\")
print(f"MC Regression & {ccs[0]['Reg']:.3f} & {ccs[1]['Reg']:.3f} & {ccs[2]['Reg']:.3f} & - & - & - \\\\")
print(f"Object Detection, 1 stage & {ccs[0]['OD1']:.3f} & {ccs[1]['OD1']:.3f} & {ccs[2]['OD1']:.3f} & {ccs[0]['OD1_10HPF']:.3f} & {ccs[1]['OD1_10HPF']:.3f} & {ccs[2]['OD1_10HPF']:.3f}  \\\\")     
print(f"Object Detection, 2 stage & {ccs[0]['OD2']:.3f} & {ccs[1]['OD2']:.3f} & {ccs[2]['OD2']:.3f} & {ccs[0]['OD2_10HPF']:.3f} & {ccs[1]['OD2_10HPF']:.3f} & {ccs[2]['OD2_10HPF']:.3f}  \\\\")     
