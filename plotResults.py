"""

   Slide-dependent statistics for estimation of High-Power-Field validity

   M. Aubreville,
   Pattern Recognition Lab, FAU Erlangen-Nürnberg

"""
import matplotlib
import matplotlib.pyplot as plt 
plt.rcParams['image.composite_image'] = False
import sqlite3
from SlideRunner.general.dependencies import *
import numpy as np
from SlideRunner.dataAccess import *
from SlideRunner.dataAccess import *
from SlideRunner.dataAccess import annotations
import SlideRunner.dataAccess.annotations
import os
import openslide
import sys
import pickle

from lib.retrieveDensity import *

PROC_ZOOM_LEVELS = [1,2,4,8,16,32]

# We did a post-training selection on the validation set, and selected amongst the last 20 epochs the
# one with the smallest error (regression approach) or highest F1 score (unet approach)
epochs_selected_dirreg = [138, 146, 149]
epochs_selected_unet = [135, 143, 132]

# Slide lists of all three validation runs
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


def plt_rect_around(center, d, linestyle, color, linewidth, label):
        rect = plt.Rectangle((center[0]-d[0]*0.5,center[1]-d[1]*0.5),d[0],d[1],linestyle=linestyle, linewidth=linewidth,edgecolor=color,facecolor='none', label=label)
        # Add the patch to the Axes
        plt.gca().add_patch(rect)

def getExpertRegion(dbase, expertID, slidename, countMap):
        """
                Retrieve the expert region from the database and calculate the respective count
        """

        slideid=dbase.findSlideWithFilename(slidename,'')
        dbase.loadIntoMemory(slideid)
        for anno in dbase.annotations.keys():
                if (dbase.annotations[anno].labels[0].annnotatorId != expertID):
                        # only take the ones from the right expert
                        continue
                if (dbase.annotations[anno].annotationType == annotations.AnnotationType.AREA) and (dbase.annotations[anno].width()>6000):
                        center = dbase.annotations[anno].centerCoordinate()
                        ds=32
                        center = [int(center.x/ds), int(center.y/ds)]
                        countVal = countMap[center[1], center[0]]

                        return round(countVal), (center[0],center[1]), countMap
        
        print('No annotation found for: ',slidename)
        return 0, (0,0), countMap

DB = Database().open('databases/MITOS_WSI_CCMCT_ODAEL.sqlite')

dbExpert = Database()
dbTumorzone = Database()
dbTumorzone.open('databases/MITOS_WSI_CCMCT_Tumorzone.sqlite')
dbExpert.open('databases/pathologists_merged.sqlite')

allSlides = dbTumorzone.execute('SELECT filename FROM Slides').fetchall()
results_1stage = []
results_2stage = []
results_1stage_10hpf = []
results_2stage_10hpf = []

slidedir = './WSI/'
resultsdir = 'results/'

print('Loading OD 10HPF results ..')
for x in range(3):
        print('... ', files_1stage_10hpf[x])
        results_1stage_10hpf.append(pickle.load(open(resultsdir+files_1stage_10hpf[x],'rb')))

print('Loading OD2 10HPF results ..')
for x in range(3):
        print('... ', files_2stage_10hpf[x])
        results_2stage_10hpf.append(pickle.load(open(resultsdir+files_2stage_10hpf[x],'rb')))

print('Loading OD results ..')
for x in range(3):
        print('... ', files_1stage[x])
        results_1stage.append(pickle.load(open(resultsdir+files_1stage[x],'rb')))

print('Loading 2nd stage OD results ..')
for x in range(3):
        results_2stage.append(pickle.load(open(resultsdir+files_2ndstage[x],'rb')))
        print('... ', files_2ndstage[x])

import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

cmap = pl.cm.Greens

my_cmap = cmap(np.arange(cmap.N))

my_cmap[:,-1] = np.linspace(0, 1, cmap.N)

my_cmap = ListedColormap(my_cmap)

cmap = pl.cm.winter

my_cmap2 = cmap(np.arange(cmap.N))

my_cmap2[:,-1] = np.linspace(0, 1, cmap.N)

my_cmap2 = ListedColormap(my_cmap2)


bp=list()
bplabel=list()
percMalign=list()
expertRegionCount=list()
for slidenumber,slidename in enumerate(testslides):
    valrun=None
    for k in range(3):
        if (slidename in slidelist[k]):
            valrun=k
            print('Part of ValRun: ',valrun)
    if valrun is None:
            continue

    countVal = dict()
    slidebasename = slidename[0:-4]

    countMap,W_x,W_y, mitosisMap, circleMap,mitlist = getMitoticCountGT(DB, slidedir, slidename)

    allExperts = dbExpert.execute('SELECT uid,name FROM Persons').fetchall()
    
    countVal=dict()
    center=dict()
    for expert,name in allExperts:
            
            countVal[name], center[name], countMap = getExpertRegion(dbExpert, expert, slidename, countMap)


    print('  Getting thumbnail image ...')
    sl = openslide.open_slide('WSI/'+slidename)
    thumb = sl.get_thumbnail(size=[countMap.shape[1], countMap.shape[0]])
    thumb = np.asarray(thumb)

    print('  Getting valid mask ...')
    validMask = retrieveValidMask('WSI/'+slidename)
    activeCountMap = countMap[validMask>0] 
    

    print('  Getting OD1 results ...')
    resultMap_OD1,single_OD1 = retrieveDensity_OD(slidedir,slidename, thresholds_1stage[valrun], results_1stage[valrun])
    print('  Getting OD2 results ...')
    resultMap_OD2,single_OD2 = retrieveDensity_OD(slidedir, slidename, thresholds_2ndstage[valrun], results_2stage[valrun])
    print('  Getting OD2 results ...')
    resultMap_Res,nonSmoothed = retrieveDensity_reg(slidedir, slidename, resultsdir, '_results_dirreg_%d.npz' % epochs_selected_dirreg[valrun])
    print('  Getting UNET results ...')
    resultMap_UNET, nonSmoothed = retrieveDensityUNET(slidedir, slidename, resultsdir, '.unet_roi_epoch%d_UNET.npz' % epochs_selected_unet[valrun])

    print('  Getting OD1 10HPF results ...')
    resultMap_OD1_10HPF,single_OD1_10HPF = retrieveDensity_OD(slidedir, slidename, thresholds_10hpf_1ststage[valrun], results_1stage_10hpf[valrun])
    print('  Getting OD2 10HPF results ...')
    resultMap_OD2_10HPF,single_OD2_10HPF = retrieveDensity_OD(slidedir, slidename, thresholds_10hpf_2ndstage[valrun], results_2stage_10hpf[valrun])

    # Apply valid mask on all results
    resultMap_UNET[validMask==0]=0
    resultMap_OD1[validMask==0]=0
    resultMap_OD2[validMask==0]=0
    resultMap_OD1_10HPF[validMask==0]=0
    resultMap_OD2_10HPF[validMask==0]=0
    resultMap_Res[cv2.resize(validMask,dsize=resultMap_Res.shape[::-1])==0]=0
    maxC_OD1 = np.unravel_index(np.argmax(resultMap_OD1, axis=None), resultMap_OD1.shape)
    countVal['AlgoOD1'] = round(countMap[maxC_OD1[0],maxC_OD1[1]])
    maxC_OD2 = np.unravel_index(np.argmax(resultMap_OD2, axis=None), resultMap_OD2.shape)
    countVal['AlgoOD2'] = round(countMap[maxC_OD2[0],maxC_OD2[1]])

    maxC_OD1_10HPF = np.unravel_index(np.argmax(resultMap_OD1_10HPF, axis=None), resultMap_OD1_10HPF.shape)
    countVal['AlgoOD1_10HPF'] = round(countMap[maxC_OD1_10HPF[0],maxC_OD1_10HPF[1]])
    maxC_OD2_10HPF = np.unravel_index(np.argmax(resultMap_OD2_10HPF, axis=None), resultMap_OD2_10HPF.shape)
    countVal['AlgoOD2_10HPF'] = round(countMap[maxC_OD2_10HPF[0],maxC_OD2_10HPF[1]])

    ds_resnet = 16
    maxC_Reg = np.unravel_index(np.argmax(resultMap_Res, axis=None), resultMap_Res.shape)
    maxC_Reg = [x*ds_resnet for x in maxC_Reg]
    countVal['AlgoReg'] = round(countMap[maxC_Reg[0],maxC_Reg[1]])
    print('ResMap shape: ',resultMap_Res.shape,'Real shape:',countMap.shape)
    ds_resnet = 1
    maxC_Map = np.unravel_index(np.argmax(resultMap_UNET, axis=None), resultMap_UNET.shape)
    maxC_Map = [x*ds_resnet for x in maxC_Map]
    print('resultMap_UNET shape: ',resultMap_UNET.shape,'Real shape:',countMap.shape)
    countVal['AlgoMap'] = round(countMap[maxC_Map[0],maxC_Map[1]])

    print('   -> Plotting')
    fig = plt.figure(figsize=(10,5))
    print('       (Main image)')
    plt.imshow(thumb)
    plt.axis('off')

    print('       (Countmap overlay)')
    plt.imshow(countMap, cmap=my_cmap)
    clb = plt.colorbar()
    clb.set_label(label='mitotic count (MC)',size=18)
    clb.ax.tick_params(labelsize=16) 
    plt.tight_layout()

    colors = {'RK': 'r', 
              'OK': 'r',
              'MD': 'r',
              'AS': 'r',
              'CG': 'r',
              'MF': 'b',
              'FB': 'b',
              'SM': 'b'}

    markers = {'RK': '*', 
              'OK': 's',
              'MD': 'o',
              'AS': 'P',
              'CG': 'X',
              'MF': 'd',
              'FB': 'p',
              'SM': '.'}

    labels = {'RK': 'BCVP1', 
              'OK': 'BCVP2',
              'MD': 'BCVP3',
              'CG': 'BCVP4',
              'AS': 'BCVP5',
              'MF': 'VPIT1',
              'FB': 'VPIT2',
              'SM': 'VPIT3'}

    linestyle = {'RK': '-', 
              'OK': '--',
              'MD': '-.',
              'AS': ':',
              'CG': '-',
              'MF': '--',
              'FB': '-.',
              'SM': ':'}



    for name in labels.keys():
            plt_rect_around(center[name], d=(W_x,W_y), linewidth=2, linestyle=linestyle[name], color=colors[name], label='%s, GTMC=%d' % (labels[name], countVal[name]))


    plt_rect_around((maxC_OD1[1],maxC_OD1[0]), d=(W_x,W_y), linewidth=2, linestyle=':', color='c', label='OD1, GTMC=%d' % countVal['AlgoOD1'])
    plt_rect_around((maxC_OD2[1],maxC_OD2[0]), d=(W_x,W_y), linewidth=2, linestyle='-', color='c', label='OD2, GTMC=%d' % countVal['AlgoOD2'])
    plt_rect_around((maxC_Reg[1],maxC_Reg[0]), d=(W_x,W_y), linewidth=2, linestyle='-.', color='c', label='Reg, GTMC=%d' % countVal['AlgoReg'])
    plt_rect_around((maxC_Map[1],maxC_Map[0]), d=(W_x,W_y), linewidth=2, linestyle='--', color='c', label='Map, GTMC=%d' % countVal['AlgoMap'])
    plt_rect_around((maxC_OD2_10HPF[1],maxC_OD2_10HPF[0]), d=(W_x,W_y), linewidth=2, linestyle='-', color='m', label='OD2-10HPF, GTMC=%d' % countVal['AlgoOD2_10HPF'])
    

    plt.legend(loc='upper right', bbox_to_anchor=(-0.05, 0.75))
    fig.subplots_adjust(left=0.35)

    print('       (saving)')
    plt.savefig('resultMap_%s.pdf' % slidename)
    plt.close(fig)
    print('       (saved)')

    fig = plt.figure(figsize=(10,5))
    plt.imshow(thumb)
    plt.axis('off')
    for name in labels.keys():
                    plt_rect_around(center[name], d=(W_x,W_y), linewidth=2, linestyle=linestyle[name], color=colors[name], label='%s, GTMC=%d' % (labels[name], countVal[name]))
        
        
    plt_rect_around((maxC_OD1[1],maxC_OD1[0]), d=(W_x,W_y), linewidth=2, linestyle=':', color='c', label='OD1, GTMC=%d' % countVal['AlgoOD1'])
    plt_rect_around((maxC_OD2[1],maxC_OD2[0]), d=(W_x,W_y), linewidth=2, linestyle='-', color='c', label='OD2, GTMC=%d' % countVal['AlgoOD2'])
    plt_rect_around((maxC_Reg[1],maxC_Reg[0]), d=(W_x,W_y), linewidth=2, linestyle='-.', color='c', label='Reg, GTMC=%d' % countVal['AlgoReg'])
    plt_rect_around((maxC_Map[1],maxC_Map[0]), d=(W_x,W_y), linewidth=2, linestyle='--', color='c', label='Seg, GTMC=%d' % countVal['AlgoMap'])
    plt_rect_around((maxC_OD2_10HPF[1],maxC_OD2_10HPF[0]), d=(W_x,W_y), linewidth=2, linestyle='-', color='m', label='OD2-10HPF, GTMC=%d' % countVal['AlgoOD2_10HPF'])

    for (x,y) in mitlist:
        plt.plot(x,y,marker='.',  color=[0,1,0], markersize=3)
        
    plt.legend(loc='upper right', bbox_to_anchor=(-0.05, 0.75))
    plt.tight_layout()
    plt.savefig('circleMap_%s.pdf' % slidename)
            
    plt.close(fig)

    expertRegionCount.append(countVal)

    percMalign += [100*np.sum(activeCountMap>=7)/activeCountMap.shape[0]]

    if (slidenumber in [1,6,10,11,12,13]):
        fig = plt.figure(figsize=(10,5))
        plt.imshow(thumb)
        plt.axis('off')
        for (x,y) in single_OD2:
                plt.plot(x,y,marker='.',  color=[0,1,0], markersize=3)

        clb.ax.tick_params(labelsize=16) 
        plt.tight_layout()
        plt_rect_around((maxC_OD2[1],maxC_OD2[0]), d=(W_x,W_y), linewidth=2, linestyle='-', color='c', label='OD2, GTMC=%d' % countVal['AlgoOD2'])
        plt.savefig('estimate_OD2%s.pdf' % slidename[:-4])
        plt.close(fig)

        fig = plt.figure(figsize=(10,5))
        plt.imshow(thumb)
        plt.axis('off')

        for (x,y) in single_OD2_10HPF:
                plt.plot(x,y,marker='.',  color=[0,1,0], markersize=3)
        plt_rect_around((maxC_OD2_10HPF[1],maxC_OD2_10HPF[0]), d=(W_x,W_y), linewidth=2, linestyle='-', color='m', label='OD2-10HPF, GTMC=%d' % countVal['AlgoOD2_10HPF'])

        plt.tight_layout()
        plt.savefig('estimate_OD2_10HPF%s.pdf' % slidename[:-4])
        plt.close(fig)

    print('Adding activeCountMap to array',activeCountMap.shape)
    bp += [activeCountMap]
    print('Done.')

# To the main results comparison plot:

# First sort into the 3 box plots

argsorted = np.argsort(percMalign)
percMalignSorted=list()
expertRegionCountSorted=list()
bpSorted=list()
for k in range(argsorted.shape[0]):
    sortIdx = argsorted[k]
    percMalignSorted.append(percMalign[sortIdx])
    bpSorted.append(bp[sortIdx])
    expertRegionCountSorted.append(expertRegionCount[sortIdx])

bp1=list()
bp1label=list()
bp2=list()
bp2label=list()
bp3=list()
bp3label=list()
OFFSET1=11
OFFSET2=21
for k in np.arange(0,OFFSET1):
    bp1 += [bpSorted[k]]
    bp1label += [k+1]

for k in np.arange(OFFSET1,OFFSET2):
    bp2 += [bpSorted[k]]
    bp2label += [k+1]

for k in np.arange(OFFSET2,32):
    bp3 += [bpSorted[k]]
    bp3label += [k+1]

import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 4


fig = plt.figure(figsize=(10,4))
sb1 = plt.subplot(1,3,1)
plt.plot(np.arange(len(bp1)+2), np.ones(len(bp1)+2)*7, 'r--')
bplot = plt.boxplot(bp1,labels=bp1label, whis='range', patch_artist=True)
for patch in bplot['boxes']:
        patch.set_facecolor('#F2F2F2')

plt.xlabel('Tumor case number')
plt.title('clearly low grade')
plt.ylabel('MC')
plt.grid(linestyle='--', color='#DDDDDD')
plt.yticks([0,2,4,6,7,8,10,12])
plotAlgos = True
for k in np.arange(0,OFFSET1):
    leg1 = plt.plot(k+0.75,expertRegionCountSorted[k]['RK'], color='r', marker='*', zorder=10)[0]
    leg2 = plt.plot(k+0.85,expertRegionCountSorted[k]['OK'], color='r', marker='s', zorder=10)[0]
    leg3 = plt.plot(k+0.95,expertRegionCountSorted[k]['CG'], color='r', marker='o', zorder=10)[0]
    leg4 = plt.plot(k+1.05,expertRegionCountSorted[k]['MD'], color='r', marker='d', zorder=10)[0]
    leg5 = plt.plot(k+0.85,expertRegionCountSorted[k]['AS'], color='r', marker='X', zorder=10)[0]
    leg6 = plt.plot(k+1.15,expertRegionCountSorted[k]['MF'], color='b', marker='P', zorder=10)[0]
    leg7 = plt.plot(k+0.85,expertRegionCountSorted[k]['FB'], color='b', marker='p', zorder=10)[0]
    leg8 = plt.plot(k+0.85,expertRegionCountSorted[k]['SM'], color='b', marker=',', zorder=10)[0]
    if (plotAlgos):
        legA = plt.plot(k+1.25,expertRegionCountSorted[k]['AlgoReg'], color='c', marker='<', zorder=10)[0]
        legD = plt.plot(k+1.05,expertRegionCountSorted[k]['AlgoMap'], color='c', marker='P', zorder=10)[0]
        legB = plt.plot(k+0.75,expertRegionCountSorted[k]['AlgoOD1'], color='c', marker='>', zorder=10)[0]
        legC = plt.plot(k+0.95,expertRegionCountSorted[k]['AlgoOD2'], color='c', marker='X', zorder=10)[0]
        legF = plt.plot(k+1.05,expertRegionCountSorted[k]['AlgoOD2_10HPF'], linewidth=0, color='m', marker='X', zorder=10)[0]

plt.subplots_adjust(left=0.05,right=0.85)

sb2 = plt.subplot(1,3,2)
plt.plot(np.arange(len(bp2)+2), np.ones(len(bp2)+2)*7, 'r--')
bplot = plt.boxplot(bp2,labels=bp2label, whis='range',patch_artist=True)
for patch in bplot['boxes']:
        patch.set_facecolor('#F2F2F2')
plt.xlabel('Tumor case number')
plt.title('borderline')
plt.yticks([0,7,25,50,75,100,125,150,175,200])
plt.ylim([-5,200])

plt.grid(linestyle='--', color='#DDDDDD')
for k in np.arange(OFFSET1,OFFSET2):
    leg1 = plt.plot(k+0.75-OFFSET1,expertRegionCountSorted[k]['RK'], color='r', marker='*', zorder=10)
    leg2 = plt.plot(k+0.85-OFFSET1,expertRegionCountSorted[k]['OK'], color='r', marker='s', zorder=10)
    leg3 = plt.plot(k+0.95-OFFSET1,expertRegionCountSorted[k]['CG'], color='r', marker='o', zorder=10)
    leg4 = plt.plot(k+1.05-OFFSET1,expertRegionCountSorted[k]['MD'], color='r', marker='d', zorder=10)
    leg5 = plt.plot(k+1.05-OFFSET1,expertRegionCountSorted[k]['AS'], color='r', marker='X', zorder=10)
    leg6 = plt.plot(k+1.15-OFFSET1,expertRegionCountSorted[k]['MF'], color='b', marker='P', zorder=10)
    leg7 = plt.plot(k+0.85-OFFSET1,expertRegionCountSorted[k]['FB'], color='b', marker='p', zorder=10)
    leg8 = plt.plot(k+0.85-OFFSET1,expertRegionCountSorted[k]['SM'], color='b', marker='.', zorder=10)
    if (plotAlgos):
        legA = plt.plot(k+1.25-OFFSET1,expertRegionCountSorted[k]['AlgoReg'], color='c', marker='<', zorder=10)
        legD = plt.plot(k+1.05-OFFSET1,expertRegionCountSorted[k]['AlgoMap'], color='c', marker='P', zorder=10)[0]
        legB = plt.plot(k+0.75-OFFSET1,expertRegionCountSorted[k]['AlgoOD1'], color='c', marker='>', zorder=10)
        legC = plt.plot(k+0.95-OFFSET1,expertRegionCountSorted[k]['AlgoOD2'], color='c', marker='X', zorder=10)[0]
        legF = plt.plot(k+1.05-OFFSET1,expertRegionCountSorted[k]['AlgoOD2_10HPF'], linewidth=0, color='m', marker='X', zorder=10)[0]

plt.yscale('symlog',linthreshy= 50)
plt.yticks([0,7,25,50,75, 100], ['0','7','25','50','75', '100'])

sb3 = plt.subplot(1,3,3)

for k in np.arange(OFFSET2,32):
    leg1 = fig.gca().plot(k+0.75-OFFSET2,expertRegionCountSorted[k]['RK'], linewidth=0, color='r', marker='*', zorder=10)[0]
    leg2 = fig.gca().plot(k+0.85-OFFSET2,expertRegionCountSorted[k]['OK'], linewidth=0, color='r', marker='s', zorder=10)[0]
    leg3 = fig.gca().plot(k+0.95-OFFSET2,expertRegionCountSorted[k]['CG'], linewidth=0, color='r', marker='o', zorder=10)[0]
    leg4 = fig.gca().plot(k+1.05-OFFSET2,expertRegionCountSorted[k]['MD'], linewidth=0, color='r', marker='d', zorder=10)[0]
    leg5 = fig.gca().plot(k+1.05-OFFSET2,expertRegionCountSorted[k]['AS'], linewidth=0, color='r', marker='X', zorder=10)[0]
    leg6 = fig.gca().plot(k+1.15-OFFSET2,expertRegionCountSorted[k]['MF'], linewidth=0, color='b', marker='P', zorder=10)[0]
    leg7 = fig.gca().plot(k+0.85-OFFSET2,expertRegionCountSorted[k]['FB'], linewidth=0, color='b', marker='p', zorder=10)[0]
    leg8 = fig.gca().plot(k+0.85-OFFSET2,expertRegionCountSorted[k]['SM'], linewidth=0, color='b', marker='.', zorder=10)[0]
    if (plotAlgos):
        legA = fig.gca().plot(k+1.25-OFFSET2,expertRegionCountSorted[k]['AlgoReg'], linewidth=0, color='c', marker='<', zorder=10)[0]
        legD = fig.gca().plot(k+1.05-OFFSET2,expertRegionCountSorted[k]['AlgoMap'], linewidth=0, color='c', marker='P', zorder=10)[0]
        legB = fig.gca().plot(k+0.75-OFFSET2,expertRegionCountSorted[k]['AlgoOD1'], linewidth=0, color='c', marker='>', zorder=10)[0]
        legC = fig.gca().plot(k+0.95-OFFSET2,expertRegionCountSorted[k]['AlgoOD2'], linewidth=0, color='c', marker='X', zorder=10)[0]
        legF = fig.gca().plot(k+1.05-OFFSET2,expertRegionCountSorted[k]['AlgoOD2_10HPF'], linewidth=0, color='m', marker='X', zorder=10)[0]

bplot = plt.boxplot(bp3,labels=bp3label, whis='range', patch_artist=True)
for patch in bplot['boxes']:
        patch.set_facecolor('#F2F2F2')

plt.plot(np.arange(len(bp3)+2), np.ones(len(bp3)+2)*7, 'r--')
plt.yticks([0,7,25,50,75,100,150,200,250,300])
plt.ylim()
plt.xlabel('Tumor case number')
plt.title('clearly high grade')
plt.grid(linestyle='--', color='#DDDDDD')


if (plotAlgos):
        plt.legend((leg1,leg2, leg3, leg4, leg5,leg6,leg7, leg8, legA,legD, legB, legC, legF),('BCVP1','BCVP2','BCVP3','BCVP4','BCVP5', 'VPIT1','VPIT2', 'VPIT3','Regression','Segmentation','OD-Stage1', 'OD-Stage2','OD2-10HPF'),
           loc='upper right', bbox_to_anchor=(1.55, 0.8))
else:
        plt.legend((leg1,leg2, leg3, leg4, leg5,leg6,leg7, leg8),('BCVP1','BCVP2','BCVP3','BCVP4','BCVP5', 'VPIT1','VPIT2', 'VPIT3'),
           loc='upper right', bbox_to_anchor=(1.55, 0.8))
plt.yscale('symlog',linthreshy= 100)
plt.yticks([0,7,25,50,75, 100,150,200], ['0','7','25','50','75', '100','150', '200'])

plt.tight_layout()
if (plotAlgos):
        plt.savefig('Region_results.eps')
        plt.savefig('Region_results.pdf')
else:
        plt.savefig('Region_results_noAlgos.eps')


for k in range(len(testslides)):
        print('ID ',k,':',testslides[argsorted[k]])
        print('   ',','.join(['%s:%d' % (exp, expertRegionCountSorted[k][exp]) for exp in expertRegionCountSorted[k].keys()]))
        