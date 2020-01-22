from SlideRunner.dataAccess import *
import openslide
import numpy as np
import cv2
import matplotlib.pyplot as plt

PROC_ZOOM_LEVELS = [1,2,4,8,16,32]


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
        if (slideid is None):
            raise Error('No slide found with slidename '+slidename)
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

