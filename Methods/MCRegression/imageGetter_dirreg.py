"""

   Image Getter for direct regression - directly regresses density of mitotic figures on slide

"""

import numpy as np
import openslide
import cv2
import os

from myTypes import *
from calc_regress import *

def mitosCount(p_x,p_y,w=512,h=512):
    d = 50

    def c_x(x):
        if (x<-d/2):
            return d  
        elif (x<d/2):
            return d/2 - x
        else:
            return 0

    return c_x(np.abs(p_x)-w/2)*c_x(np.abs(p_y)-h/2)/d/d


def findMitosesAnnotations(DBcur,leftUpper, rightLower, slideUID):
    q = ('SELECT coordinateX, coordinateY FROM Annotations_coordinates LEFT JOIN Annotations on Annotations.uid == Annotations_coordinates.annoId WHERE coordinateX >= '+str(leftUpper[0])+
            ' AND coordinateX <= '+str(rightLower[0])+' AND coordinateY >= '+str(leftUpper[1])+' AND coordinateY <= '+str(rightLower[1])+
            ' AND Annotations.slide == %d AND (type==1 ) and agreedClass==2'%(slideUID) )
    DBcur.execute(q)
    return DBcur.fetchall()

def findRandomSlide(DBcur, slideList, mitosisClass):
    if (slideList is not None):
        querySlide = 'AND (Annotations.slide == %s)' % (' OR Annotations.slide == '.join(slideList))
    else:
        querySlide = ''


    q = 'SELECT Annotations.slide, Slides.filename FROM Annotations_coordinates LEFT JOIN Annotations ON Annotations.uid == Annotations_coordinates.annoID LEFT JOIN Slides on Slides.uid == Annotations.slide WHERE agreedClass==%d  %s ORDER BY RANDOM() LIMIT 1' % (mitosisClass, querySlide)

    DBcur.execute(q)
    return DBcur.fetchone()


def findRandomAnnotation(DBcur, slide, YCoordinateThresholds:tuple, dataSource:LearningStep, slideObject=None, agreedClass=2):

    if (dataSource == LearningStep.TRAINING):
        q = 'SELECT coordinateX, coordinateY, Annotations.slide FROM Annotations_coordinates LEFT JOIN Annotations ON Annotations.uid == Annotations_coordinates.annoID WHERE agreedClass==%d  AND (Annotations.slide == %s) AND (coordinateY<= %d) ORDER BY RANDOM() LIMIT 1' % (agreedClass, slide, YCoordinateThresholds[0])
    elif (dataSource == LearningStep.TEST):
        q = 'SELECT coordinateX, coordinateY, Annotations.slide FROM Annotations_coordinates LEFT JOIN Annotations ON Annotations.uid == Annotations_coordinates.annoID WHERE agreedClass==%d  AND (Annotations.slide == %s) ORDER BY RANDOM() LIMIT 1' % (agreedClass, slide)
    elif (dataSource == LearningStep.VALIDATION):
        q = 'SELECT coordinateX, coordinateY, Annotations.slide FROM Annotations_coordinates LEFT JOIN Annotations ON Annotations.uid == Annotations_coordinates.annoID WHERE agreedClass==%d  AND (Annotations.slide == %s) AND (coordinateY> %d) ORDER BY RANDOM() LIMIT 1' % (agreedClass, slide, YCoordinateThresholds[0])
    DBcur.execute(q)
    res = DBcur.fetchone()
    if (res is not None):
        return res
    else:
#        print('Unable to find annotation on slide %d' % slide)
        x = np.random.randint(slideObject.dimensions[0])
        y = np.random.randint(slideObject.dimensions[1])
        return (x, y, slide)


def getImageBatch(batchSize, DB, slideList, width, height, rotate:bool) -> (np.ndarray,np.ndarray,np.ndarray):
    """
        Get an image batch from the whole slide image

    """
    images = np.empty((0,width,height,3))


def getMitosisClass( DBcur) -> int:
    q = 'SELECT uid from Classes WHERE upper(name) like "MITO%"'
    DBcur.execute(q)
    res = DBcur.fetchone()
    return res[0]


def getImage(DB, slideList:list,width:int,height:int, basepath:str,  
             learningStep:LearningStep, showSubRects=False, ds=32, allAreaPick=False, rescale=1.0, hardExamplePick=False) -> (list, list, list, dict):
    splitter = TrainTestSplit(train_percentage=0.8, test_percentage=0.0)
    origsize = (width,height)
    DBcur = DB.cursor()
    mitosisClass = getMitosisClass(DBcur)
    slide,slidefilename = findRandomSlide(DBcur, slideList, mitosisClass)
#    slide,slidefilename = findRandomSlide(DBcur, slideList)
    w_orig = int(width/rescale)
    h_orig = int(height/rescale)
    

    if slidefilename is not None:
        sl = openslide.open_slide(basepath + os.sep + slidefilename)

        if (sl.dimensions is None):
            print('Something went wrong with: %s' % slidefilename)

        tries=0
        while (tries<1000):
            tries+=1

            # allow for completely random picks
            if (allAreaPick == True):
                coord_x_center = np.random.randint(sl.dimensions[0])
                coord_y_center = np.random.randint(sl.dimensions[1])
            else:
                if (hardExamplePick):
                    if (np.random.rand(1)<0.5):
                        # human hard example pick: disagreed class
                        coord_x_center, coord_y_center, slide = findRandomAnnotation(DBcur, slide, YCoordinateThresholds=splitter.getYSplit(sl.dimensions[1]), dataSource=learningStep, slideObject=sl, agreedClass=0)
                    else:
                        # human hard example pick: Schrott class
                        coord_x_center, coord_y_center, slide = findRandomAnnotation(DBcur, slide, YCoordinateThresholds=splitter.getYSplit(sl.dimensions[1]), dataSource=learningStep, slideObject=sl, agreedClass=4)
                else:
                    coord_x_center, coord_y_center, slide = findRandomAnnotation(DBcur, slide, YCoordinateThresholds=splitter.getYSplit(sl.dimensions[1]), dataSource=learningStep, slideObject=sl, agreedClass=2)

                # do not always show cell with anno in the center of the slide
                coord_x_center += int((2*np.random.rand(1)-1)*w_orig*0.55)
                coord_y_center += int((2*np.random.rand(1)-1)*h_orig*0.55)

#                coord_x_center =  cx_orig + int((2*np.random.rand(1)-1)*width*0.55)
#                coord_y_center =  cy_orig + int((2*np.random.rand(1)-1)*width*0.55)


            if (learningStep == LearningStep.TRAINING):
                rotate = np.random.rand(1)*np.pi*2
                if (rotate > 0):
                    width_new = max(origsize[0],height * np.abs(np.sin(np.mod(rotate,np.pi))) + origsize[0] * np.abs(np.cos(np.mod(rotate,np.pi))))
                    height_new = max(origsize[1],width * np.abs(np.sin(np.mod(rotate,np.pi))) + origsize[1] * np.abs(np.cos(np.mod(rotate,np.pi))))
                    width = int(width_new)
                    height = int(height_new)
            else:
                rotate = 0

            # Image is valid only if within slide boundaries        
            if ((coord_x_center-(width/rescale/2)>0) and (coord_y_center-(height/rescale/2)>0) and
                (coord_x_center+(width/rescale/2)<sl.dimensions[0]) and (coord_y_center+(height/rescale/2)<sl.dimensions[1])):
                break

        images=list()

        slide_idx = np.argmin(np.abs(1-np.float32(sl.level_downsamples)))
        act_downsample=sl.level_downsamples[slide_idx]
        leftUpper = ((coord_x_center-int(width/2/rescale),(coord_y_center-int(height/2/rescale))))
        rightLower = ((coord_x_center+int(width/2/rescale),(coord_y_center+int(height/2/rescale))))

        annos = findMitosesAnnotations(DBcur,leftUpper,rightLower, slide)
        im = np.uint8(sl.read_region(location=(coord_x_center-int(width/2/rescale), coord_y_center-int(height/2/rescale)), 
                                        size=(int(width/act_downsample/rescale),int(height/act_downsample/rescale)), level=slide_idx))

        im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        im = cv2.resize(im, dsize=(width,height))

        M = cv2.getRotationMatrix2D((width/2, height/2),rotate*180/np.pi, 1)
        im = cv2.warpAffine(np.float32(im), M, (width,height))




        M_small = cv2.getRotationMatrix2D((0,0),rotate*180/np.pi, 1)
#        mapImage_small = np.zeros(shape=(1,int(origsize[1]/ds),int(origsize[0]/ds),1), dtype=np.float32)
        annoList = np.empty((0,3), dtype=np.int32)
        cnt = 0

        for anno in annos:
            newCoords = np.asarray(((anno[0]-leftUpper[0]-width/2/rescale)*rescale, (anno[1]-leftUpper[1]-height/2/rescale)*rescale,1))
            center_trans = np.int16(np.matmul(np.reshape(newCoords,(1,3)),M_small.T))

            cnt += mitosCount(center_trans[0][0], center_trans[0][1], origsize[0], origsize[1])                

        im = cv2.getRectSubPix(im, origsize, (width/2,height/2))


    return np.uint8(im), cnt/10 # can't ever be more than 10 mitoses / image
