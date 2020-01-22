import numpy as np
import openslide
import cv2
import os
import math

from myTypes import *
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out




def findSpotAnnotations(DBcur,leftUpper, rightLower, slideUID):
    q = ('SELECT coordinateX, coordinateY FROM Annotations_coordinates LEFT JOIN Annotations on Annotations.uid == Annotations_coordinates.annoId WHERE coordinateX >= '+str(leftUpper[0])+
            ' AND coordinateX <= '+str(rightLower[0])+' AND coordinateY >= '+str(leftUpper[1])+' AND coordinateY <= '+str(rightLower[1])+
            ' AND Annotations.slide == %d AND (type==1 ) and agreedClass==2'%(slideUID) )
    DBcur.execute(q)
    return DBcur.fetchall()

def findRandomMitosisSlide(DBcur, slideList):

    querySlide = ' OR Annotations.slide == '.join(slideList)

    q = 'SELECT Annotations.slide, Slides.filename FROM Annotations_coordinates LEFT JOIN Annotations ON Annotations.uid == Annotations_coordinates.annoID LEFT JOIN Slides on Slides.uid == Annotations.slide WHERE agreedClass==2  AND (Annotations.slide == %s) ORDER BY RANDOM() LIMIT 1' % (querySlide)

    DBcur.execute(q)
    return DBcur.fetchone()

def findRandomSlide(DBcur, slideList):

    querySlide = ','.join(slideList)

    q = 'SELECT Slides.uid, Slides.filename FROM Slides WHERE Slides.uid in (%s) ORDER BY RANDOM() LIMIT 1' % (querySlide)

    DBcur.execute(q)
    return DBcur.fetchone()


def findRandomAnnotation(DBcur, slide, YCoordinateThresholds:tuple, dataSource:LearningStep, slideObject=None, agreedClass=2):

    q = 'SELECT coordinateX, coordinateY, Annotations.slide FROM Annotations_coordinates LEFT JOIN Annotations ON Annotations.uid == Annotations_coordinates.annoID WHERE agreedClass==%d  AND (Annotations.slide == %s) ORDER BY RANDOM() LIMIT 1' % (agreedClass, slide)

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


def getImage(DB, slideList:list,width:int,height:int, basepath:str,  
             learningStep:LearningStep, showSubRects=False, allAreaPick=False, hardExamplePick=False, timerObject = None, normalize = False) -> (list, list, list, dict):
    splitter = TrainTestSplit(train_percentage=0.8, test_percentage=0.0)
    origsize = (width,height)
    DBcur = DB.cursor()
    if (allAreaPick):
        slide,slidefilename = findRandomSlide(DBcur, slideList)
    else:
        slide,slidefilename = findRandomMitosisSlide(DBcur, slideList)

    if slidefilename is not None:
        if (timerObject):
            openSlideTimerId = timerObject.startTimer('OPENSLIDE')
        sl = openslide.open_slide(basepath + os.sep + slidefilename)
        if (timerObject):
            timerObject.stopTimer('OPENSLIDE',openSlideTimerId)
        if (timerObject):
            DBTimerID = timerObject.startTimer('DB')

        if (sl.dimensions is None):
            print('Something went wrong with: %s' % slidefilename)

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
            coord_x_center += int(np.random.rand(1)*width*0.45)
            coord_y_center += int(np.random.rand(1)*width*0.45)

#        DBcur.execute('SELECT filename from Slides WHERE uid==%d' % slide)

        if (learningStep == LearningStep.TRAINING):
            rotate = np.random.rand(1)*np.pi*2
            if (rotate > 0):
                width_new = max(width,height * np.abs(np.sin(np.mod(rotate,np.pi))) + width * np.abs(np.cos(np.mod(rotate,np.pi))))
                height_new = max(height,width * np.abs(np.sin(np.mod(rotate,np.pi))) + height * np.abs(np.cos(np.mod(rotate,np.pi))))
                width = int(width_new)
                height = int(height_new)
        else:
            rotate = 0

        images=list()
        maps=list()
        mapUpscaled = list()

        idx = 0
        downsample = 1
        slide_idx = np.argmin(np.abs(downsample-np.float32(sl.level_downsamples)))
        act_downsample=sl.level_downsamples[slide_idx]
        leftUpper = ((coord_x_center-int(width*downsample/2),(coord_y_center-int(height*downsample/2))))
        rightLower = ((coord_x_center+int(width*downsample/2),(coord_y_center+int(height*downsample/2))))
        annos = findSpotAnnotations(DBcur,leftUpper,rightLower, slide)
        if (timerObject):
            timerObject.stopTimer('DB',DBTimerID)


        if (timerObject):
            openSlideTimerId = timerObject.startTimer('OPENSLIDE')
        im = np.uint8(sl.read_region(location=(coord_x_center-int(width*downsample/2), coord_y_center-int(height*downsample/2)), 
                                        size=(int(width*downsample/act_downsample),int(height*downsample/act_downsample)), level=slide_idx))
        if (timerObject):
            timerObject.stopTimer('OPENSLIDE',openSlideTimerId)

        if (timerObject):
            annoTimerId = timerObject.startTimer('ANNOTATE')

        im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        im = cv2.resize(im, dsize=(width,height))
#        cv2.imwrite(filename='ds_%d.tif'%idx, img=im)

        mapImage = np.zeros(shape=(1, height, width, 1), dtype=np.float32)

        for anno in annos:
            center=(origsize[0]/2, origsize[1]/2)
            dscale = downsample
            pts1=(int(center[0]-origsize[0]/2/dscale),int(center[1]-origsize[1]/2/dscale))

            center = (int(32*(anno[0]-leftUpper[0])/downsample), int(32*(anno[1]-leftUpper[1])/downsample))
            radius = int(25/downsample*32)
            cv2.circle(mapImage[0], center=center, color=255, radius=max(32*3,radius), shift=5, lineType=-1, thickness=-1)

#           cv2.imwrite(filename='mi_%d.tiff'%idx, img=np.uint8(mapImage[0]))
        
#          kernel = cv2.getGaussianKernel(ksize=151, sigma=30)

#            cv2.imwrite(filename='mib_%d.tiff'%idx, img=np.uint8(mapImage[0]))

        M = cv2.getRotationMatrix2D((width/2, height/2),rotate*180/np.pi, 1)
        im = cv2.warpAffine(np.float32(im), M, (width,height))
        mapImage[0,:,:,0] = cv2.warpAffine(np.float32(mapImage[0]), M, (width,height))
        
        im = cv2.getRectSubPix(im, origsize, (width/2,height/2))
        mapImage = np.reshape(cv2.getRectSubPix(mapImage[0], origsize, (width/2,height/2)),(1,origsize[1],origsize[0],1))

        maps.append(np.float32(mapImage)*2)

        images.append(np.reshape(np.uint8(im), (1,origsize[1],origsize[0],3)))
        if (timerObject):
            timerObject.stopTimer('ANNOTATE', annoTimerId)


    return images[0], maps[0], {'slide':slide, 'x_center':coord_x_center, 'y_center':coord_y_center}
