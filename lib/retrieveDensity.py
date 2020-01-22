import openslide
import numpy as np
import cv2
import os
import bz2

def retrieveDensity_OD(slidedir:str, fname : str, threshold:float, results:dict) -> (np.ndarray,list):
    """ 

    Retrieve mitotic density from Object Detection approach results
    
    """
    sl = openslide.open_slide(slidedir+os.sep+fname)
    ds=32
    downsampledMap = np.zeros((int(sl.dimensions[1]/ds), int(sl.dimensions[0]/ds)))
    A = 2.37 # mm^2
    W_hpf_microns = np.sqrt(A*4/3) * 1000 # in microns
    H_hpf_microns = np.sqrt(A*3/4) * 1000 # in microns

    micronsPerPixel = sl.properties[openslide.PROPERTY_NAME_MPP_X]

    W_hpf = int(W_hpf_microns / float(micronsPerPixel))
    H_hpf = int(H_hpf_microns / float(micronsPerPixel))

    W_x = int(W_hpf / ds)
    W_y = int(H_hpf / ds)
    dets = []
    kernel = np.ones((W_y,W_x),np.float32)
        
    for idx in range(len(results[fname])):
            row = results[fname][idx]
            if (row[5]>threshold):
                downsampledMap[int((row[1]+row[3])/2/ds),int((row[0]+row[2])/2/ds)] += 1
                dets.append((int((row[0]+row[2])/2/ds),int((row[1]+row[3])/2/ds)))
    mitoticCount = cv2.filter2D(downsampledMap, -1, kernel ) 

    return mitoticCount, dets


def retrieveDensity_reg(slidedir:str, filename : str, resultsdir : str, suffix : str = '_results_dirreg.npz'):
        """ 

        Retrieve mitotic density from Regression approach results
        
        """
        TILESIZE_X = 512
        TILESIZE_Y = 512
        sl = openslide.open_slide(slidedir+os.sep+filename)

        tiles_total_x = int(np.floor(sl.dimensions[0] / TILESIZE_X))
        tiles_total_y = int(np.floor(sl.dimensions[1] / TILESIZE_Y))

        # calculate 10 HPFs with highest mitotic activity
        # 1 HPF = 0.237 mm^2 
        A = 2.37 # mm^2 
        W_hpf_microns = np.sqrt(A*4/3) * 1000 # in microns
        H_hpf_microns = np.sqrt(A*3/4) * 1000 # in microns

        micronsPerPixel = sl.properties[openslide.PROPERTY_NAME_MPP_X]

        W_hpf = int(W_hpf_microns / float(micronsPerPixel))  
        H_hpf = int(H_hpf_microns / float(micronsPerPixel))

        W_x = int(W_hpf / TILESIZE_X)
        W_y = int(H_hpf / TILESIZE_Y)

        f = np.load(bz2.BZ2File(resultsdir + os.sep + filename + suffix+'.bz2','rb'))
        

        scorefield=np.zeros((np.max(f['tilesProcessed'][:,1])+1,1+np.max(f['tilesProcessed'][:,0])))
        scorefield[f['tilesProcessed'][:,1],f['tilesProcessed'][:,0]] = np.reshape(f['scores'],-1)

        completeMap = scorefield

        kernel = np.ones((W_y,W_x),np.float32)
        ma = cv2.filter2D(completeMap, -1, kernel )

        return ma, completeMap


def retrieveDensityUNET(slidedir:str,filename : str, resultsdir:str, suffix : str):
        """ 

        Retrieve mitotic density from Segmentation approach results
        
        """
        sl = openslide.open_slide(slidedir+os.sep+filename)
        ds = 32
        # calculate 10 HPFs with highest mitotic activity
        # 1 HPF = 0.237 mm^2 
        A = 2.37 # mm^2 
        W_hpf_microns = np.sqrt(A*4/3) * 1000 # in microns
        H_hpf_microns = np.sqrt(A*3/4) * 1000 # in microns

        micronsPerPixel = sl.properties[openslide.PROPERTY_NAME_MPP_X]

        W_hpf = int(W_hpf_microns / float(micronsPerPixel))  
        H_hpf = int(H_hpf_microns / float(micronsPerPixel))

        W_x = int(W_hpf / ds)
        W_y = int(H_hpf / ds)

        f = np.load(bz2.BZ2File(resultsdir + os.sep + filename + suffix+'.bz2','rb'))


        completeMap = f['imageMap']

        kernel = np.ones((W_y,W_x),np.float32)
        ma = cv2.filter2D(completeMap, -1, kernel )

        return ma, completeMap

def getMitoticCountGT(DB, slidedir:str, fname:str,ds=32):
    """
        Retrieve ground truth mitotic count

    """
    sl = openslide.open_slide(slidedir+fname)
    downsampledMapGT = np.zeros((int(sl.dimensions[1]/ds), int(sl.dimensions[0]/ds)))
    uid = DB.findSlideWithFilename(fname,'')
    downsampledMapCirc = np.zeros((int(sl.dimensions[1]/ds), int(sl.dimensions[0]/ds)))
    DB.loadIntoMemory(uid)
    mitlist=[]
    for anno in DB.annotations:
        if (DB.annotations[anno].agreedClass==2):
            annodet = DB.annotations[anno]
            downsampledMapGT[int(annodet.y1/ds),int(annodet.x1/ds)] += 1
            cv2.circle(downsampledMapCirc, center=(int(annodet.x1/ds),int(annodet.y1/ds)), radius=10, thickness=-1, color=[255])
            mitlist.append((int(annodet.x1/ds),int(annodet.y1/ds)))
    A = 2.37 # mm^2
    W_hpf_microns = np.sqrt(A*4/3) * 1000 # in microns
    H_hpf_microns = np.sqrt(A*3/4) * 1000 # in microns

    micronsPerPixel = sl.properties[openslide.PROPERTY_NAME_MPP_X]

    W_hpf = int(W_hpf_microns / float(micronsPerPixel))
    H_hpf = int(H_hpf_microns / float(micronsPerPixel))

    W_x = int(W_hpf / ds)
    W_y = int(H_hpf / ds)
    kernel = np.ones((W_y,W_x),np.float32)
    mitoticCountGT = cv2.filter2D(downsampledMapGT, -1, kernel ) 
        
    return mitoticCountGT, W_x, W_y, downsampledMapGT, downsampledMapCirc, mitlist
