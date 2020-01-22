"""
   Select the best UNET model in the last available 10 epochs for Mitotic Figure ROI detection task

   With loads of inspiration from: https://github.com/kkweon/UNet-in-Tensorflow

   Using the compatibility mode of TensorFlow 2.x

Loss function: maximize IOU

    (intersection of prediction & ground truth)
    -------------------------------
    (union of prediction & ground truth)

Original Paper:
    https://arxiv.org/abs/1505.04597
"""


import time
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import sqlite3
import openslide
import cv2
import queue
import imageGetter
import threading
import sys
from queue import Queue
import platform
import matplotlib
matplotlib.use('Agg')
from myTypes import *

class timer():
    def __init__(self):
        self.totalcount = 0
        self.totaltime = 0
        self.lastTimestamps = dict()
    
    def hits(self):
        return self.totalcount
    
    def averagetime(self):
        return (self.totaltime / (self.totalcount+1E-6))

    def start(self):
        key = np.random.randint(100000)
        self.lastTimestamps[key] = time.time()
        return key
    
    def summary(self):
        return '%.2f s (%d hits)' % (self.averagetime(), self.hits())
    
    def stop(self,key):
        if (key in self.lastTimestamps):
            self.totalcount += 1
            self.totaltime += time.time() - self.lastTimestamps[key]
            del self.lastTimestamps[key]

class timerObject():
    def __init__(self):
        self.timers = dict()
    
    def addTimer(self, name:str):
        self.timers[name] = timer()
    
    def startTimer(self, name:str):
        return self.timers[name].start()
    
    def stopTimer(self, name:str, key):
        self.timers[name].stop(key)
    
    def summary(self):
        str = ''
        for tmr in self.timers.keys():
            str += tmr + ' : ' + self.timers[tmr].summary()+' '
        
        print(time.strftime("%d.%m %H:%M:%S"),str)


TIMER = timerObject()
TIMER.addTimer('TF')
TIMER.addTimer('GETIMAGE')
TIMER.addTimer('OPENSLIDE')
TIMER.addTimer('ANNOTATE')
TIMER.addTimer('DB')

multiThreaded = True

basepath = '../../MITOS_WSI_CCMCT/WSI/'
dbpath = '../../MITOS_WSI_CCMCT/databases/'

slidelist = ['26', '12', '23', '13', '28', '24', '29', '25', '11', '27', '30', '31', '4', '18', '20', '30',
                   '14', '15', '32', '6', '7', '8', '22', '19', '34', '17', '21', '35','36', '1', '2', '3' ,'9']

slidelist = np.unique(slidelist).tolist()

slidelist_test_1 = ['27', '30', '31', '6', '18', '20', '1', '2', '3' ,'9', '11']
slidelist_test_2 = ['26', '12', '23', '19', '34', '17', '24', '29', '25', '7', '4']
slidelist_test_3 = ['13', '28', '14', '15', '32', '21', '35','36', '8', '22']

if (sys.argv[1] == '1'):
    slidelist_test = slidelist_test_1
elif (sys.argv[1] == '2'):
    slidelist_test = slidelist_test_2
elif (sys.argv[1] == '3'):
    slidelist_test = slidelist_test_3

tag = 'unet_roi_odael_'+sys.argv[1]


slidelist_train = np.array(slidelist)[np.where([slidelist[x] not in slidelist_test for x in range(len(slidelist))])[0]].tolist()


PARALLEL_GETTERS = 6
IMAGESIZE_X = 384
IMAGESIZE_Y = 384

batchsize=6
def getImageBatch(lstep, DB):
        global TIMER
        im = np.zeros(shape=(batchsize,IMAGESIZE_Y,IMAGESIZE_X,3), dtype=np.uint8)
        out = np.zeros(shape=(batchsize,IMAGESIZE_Y,IMAGESIZE_X,1), dtype=np.float32)

        for k in range(batchsize):
            r = np.random.rand(1)
            if (lstep == LearningStep.TRAINING) or (lstep==LearningStep.VALIDATION):
                if (r<0.33):                        
                    im[k], out[k],foo = imageGetter.getImage(DB, slidelist_train, IMAGESIZE_X, IMAGESIZE_Y, learningStep=lstep,basepath=basepath, timerObject=TIMER)
                elif (r<0.66):
                    im[k], out[k],foo = imageGetter.getImage(DB, slidelist_train, IMAGESIZE_X, IMAGESIZE_Y, learningStep=lstep,basepath=basepath,allAreaPick=True, timerObject=TIMER)
                else:
                    im[k], out[k],foo = imageGetter.getImage(DB, slidelist_train, IMAGESIZE_X, IMAGESIZE_Y, learningStep=lstep,basepath=basepath,hardExamplePick=True, timerObject=TIMER)
            else: # test
                im[k], out[k],foo = imageGetter.getImage(DB, slidelist_test, IMAGESIZE_X, IMAGESIZE_Y, learningStep=lstep,basepath=basepath, allAreaPick=True, timerObject=TIMER)
        
        return (im,np.float32(out)/255.0/2,lstep)


def getFromQueue(queue, requestQueue):
        global dbpath
        print('DB is: ',dbpath + os.sep +  'MITOS_WSI_CCMCT_ODAEL.sqlite')
        DB = sqlite3.connect(dbpath + os.sep + 'MITOS_WSI_CCMCT_ODAEL.sqlite')
        while (True):
            lstep = requestQueue.get()
            queue.put(getImageBatch(lstep, DB))

def getImageFromQueue(queue, requestQueue, DBase):
    getImTimerId = TIMER.startTimer('GETIMAGE')
    if (multiThreaded):
        retval = queue.get()
    else:
        lstep = requestQueue.get()
        retval = getImageBatch(lstep, DBase)
    TIMER.stopTimer('GETIMAGE', getImTimerId)
    return retval



def conv_conv_pool(input_, n_filters, training, name, pool=True, activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(net, F, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1), momentum=0.9, epsilon=1e-2)
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upsample_concat(inputA, input_B, name):
    """Upsample `inputA` and concat with `input_B`

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    upsample = upsampling_2D(inputA, size=(2, 2), name=name)

    return tf.concat([upsample, input_B], axis=-1, name="concat_{}".format(name))


def upsampling_2D(tensor, name, size=(2, 2)):
    """Upsample/Rescale `tensor` by size

    Args:
        tensor (4-D Tensor): (N, H, W, C)
        name (str): name of upsampling operations
        size (tuple, optional): (height_multiplier, width_multiplier)
            (default: (2, 2))

    Returns:
        output (4-D Tensor): (N, h_multiplier * H, w_multiplier * W, C)
    """
    H, W, _ = tensor.get_shape().as_list()[1:]

    H_multi, W_multi = size
    target_H = H * H_multi
    target_W = W * W_multi

    return tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name="upsample_{}".format(name))


def make_unet(Xhinted, training):
    """Build a U-Net architecture

    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    net = Xhinted / 127.5 - 1
    net = tf.layers.conv2d(net, 4, (1, 1), name="color_space_adjust")
    conv1, pool1 = conv_conv_pool(net, [8, 8], training, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, name=4)
    conv5 = conv_conv_pool(pool4, [128, 128], training, name=5, pool=False)

    up6 = upsample_concat(conv5, conv4, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, name=6, pool=False)

    up7 = upsample_concat(conv6, conv3, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, name=7, pool=False)

    up8 = upsample_concat(conv7, conv2, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, name=8, pool=False)

    up9 = upsample_concat(conv8, conv1, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, name=9, pool=False)

    return tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=tf.nn.sigmoid, padding='same')


def SensSpec(y_pred, y_true, lambda_value):
    """ Returns an Sensitivity/Specificity loss, as described by Brosch et al.

        Inspired by the paper from Sudre et al
    """

    with tf.name_scope('Loss'):
        H, W, _ = y_pred.get_shape().as_list()[1:]

        pred_flat = tf.reshape(y_pred, [-1, H * W])
        true_flat = tf.reshape(y_true, [-1, H * W])

        with tf.name_scope('Sens'):
            with tf.name_scope('Sens_error'):
                sens_error_term = (lambda_value) * tf.reduce_sum(tf.square(true_flat - pred_flat) * true_flat, axis=1) / (tf.reduce_sum(true_flat,axis=1) + 1e-7)
            with tf.name_scope('Sensitivity'):
                sens_term = tf.reduce_sum((pred_flat) * true_flat, axis=1) / (tf.reduce_sum(true_flat,axis=1) + 1e-7)

        with tf.name_scope('Spec'):
            with tf.name_scope('Spec_error'):
                spec_error_term = (1-lambda_value) * tf.reduce_sum(tf.square(true_flat - pred_flat) * (1-true_flat), axis=1) / (tf.reduce_sum(1 - true_flat, axis=1) + 1e-7)
            with tf.name_scope('Specificity'):
                spec_term = tf.reduce_sum((1-pred_flat) * (1-true_flat), axis=1) / (tf.reduce_sum((1-true_flat),axis=1) + 1e-7)

    return  tf.add(sens_error_term,spec_error_term, name='Loss_addition'), sens_term, spec_term


def IOU_(y_pred, y_true):
    """Returns a (approx) IOU score (Jaccard index)

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7

    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)

    Returns:
        float: IOU score
    """
    with tf.name_scope('IOU'):
        H, W, _ = y_pred.get_shape().as_list()[1:]

        pred_flat = tf.reshape(y_pred, [-1, H * W])
        true_flat = tf.reshape(y_true, [-1, H * W])

        intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
        denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7

        return tf.reduce_mean(intersection / denominator)


def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.
    - width: desired width of grid/output. Used
      to downsample or upsample.
    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.
    Returns
    -------
    - normalized gird (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    # grab batch size
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to (x_t, y_t , 1)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])
    # batch_grids = tf.transpose(batch_grids, [0, 2, 1, 3])

    return batch_grids

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - interpolated images according to grids. Same size as grid.
    """
    # prepare useful params
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[3]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # cast indices as float32 (for rescaling)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # rescale x and y to [0, W/H]
    x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def make_train_op(y_pred, y_true, var_list = None):
    """Returns a training operation

    Loss function = - IOU(y_pred, y_true)

    IOU is

        (the area of intersection)
        --------------------------
        (the area of two boxes)

    Args:
        y_pred (4-D Tensor): (N, H, W, 1)
        y_true (4-D Tensor): (N, H, W, 1)

    Returns:
        train_op: minimize operation
    """
#    loss = -IOU_(y_pred, y_true)
    loss = -realIOU(y_pred, y_true)
 #   loss = tf.reduce_mean(tf.square(y_pred-y_true))
#    loss,sens,spec = SensSpec(y_pred,y_true,0.1)
 
    global_step = tf.train.get_or_create_global_step()

    optim = tf.train.AdamOptimizer()
#    optim = tf.train.GradientDescentOptimizer()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if (var_list is not None):
            return optim.minimize(loss, global_step=global_step, var_list=var_list)
        else:
            return optim.minimize(loss, global_step=global_step)

def scale_centerpiece(net, scalingFactor, batchsize=2):

    theta = np.reshape(np.asarray([[1/scalingFactor,0,0],[0,1/scalingFactor,0]]),[1, 2, 3])
    theta = np.repeat(theta,repeats=batchsize,axis=0)

    theta_tensor = tf.constant(theta)
    batch_grids = affine_grid_generator(IMAGESIZE_Y,IMAGESIZE_X, theta_tensor)

    # extract x and y coordinates
    x_s = tf.squeeze(batch_grids[:, 0:1, :, :])
    y_s = tf.squeeze(batch_grids[:, 1:2, :, :])

    # sample input with grid to get output
    out_fmap = bilinear_sampler(net, x_s, y_s)

    return out_fmap


def read_flags():
    """Returns flags"""

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('xval', type=int)

    parser.add_argument('modeldir', type=str)

    parser.add_argument("--epochs",
                        default=150,
                        type=int,
                        help="Number of epochs (default: 32)")

    parser.add_argument("--startepoch",
                        default=0,
                        type=int,
                        help="Starting index for epoch")

    parser.add_argument("--batch-size",
                        default=4,
                        type=int,
                        help="Batch size (default: 4)")

    parser.add_argument("--logdir",
                        default="logdir",
                        help="Tensorboard log directory (default: logdir)")

    flags = parser.parse_args()
    return flags

def findSpotAnnotations(DBcur,leftUpper, rightLower, slideUID):
    q = ('SELECT coordinateX, coordinateY FROM Annotations_coordinates LEFT JOIN Annotations on Annotations.uid == Annotations_coordinates.annoId WHERE coordinateX >= '+str(leftUpper[0])+
            ' AND coordinateX <= '+str(rightLower[0])+' AND coordinateY >= '+str(leftUpper[1])+' AND coordinateY <= '+str(rightLower[1])+
            ' AND Annotations.slide == %d AND (type==1 ) and agreedClass==2'%(slideUID) )
    print(q)
    DBcur.execute(q)
    return DBcur.fetchall()

def findRandomAnnotation(DBcur, slideList):

    querySlide = ' OR Annotations.slide == '.join(slideList)

    q = 'SELECT coordinateX, coordinateY, Annotations.slide FROM Annotations_coordinates LEFT JOIN Annotations ON Annotations.uid == Annotations_coordinates.annoID WHERE agreedClass==2  AND (Annotations.slide == %s) ORDER BY RANDOM() LIMIT 1' % (querySlide)
    print(q)

    DBcur.execute(q)
    return DBcur.fetchone()


def realIOU(y_pred, y_true):
    """Returns a IOU score (dice coefficient)

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = intersection / ((y_pred + y_true - intersection).sum() + 1e-7) + 1e-7

    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)

    Returns:
        float: IOU score
    """
    with tf.name_scope('IOU'):
        H, W, _ = y_pred.get_shape().as_list()[1:]

        pred_flat = tf.reshape(y_pred, [-1, H * W])
        true_flat = tf.reshape(y_true, [-1, H * W])

        intersection = tf.reduce_sum(pred_flat * true_flat, axis=1)
        denominator = tf.reduce_sum(pred_flat+true_flat - pred_flat * true_flat, axis=1)  + 1e-7

        return tf.reduce_mean(intersection / denominator)


def main(flags):

    getterQueue = Queue()
    requestQueue = Queue()


    if (multiThreaded):
        procs = list()
        for idx in range(PARALLEL_GETTERS):
            proc = threading.Thread(target=getFromQueue, args=(getterQueue,requestQueue,))
            proc.daemon=True
            proc.start()
            procs.append(proc)

    print('DB is: ',dbpath + os.sep +  'MITOS_WSI_CCMCT_ODAEL.sqlite')
    DB = sqlite3.connect(dbpath + os.sep +  'MITOS_WSI_CCMCT_ODAEL.sqlite')
    DBcur = DB.cursor()


    current_time = time.strftime("%m_%d_%H_%M_%S")
    train_logdir = os.path.join(flags.logdir, "train"+tag, current_time)
    val_logdir = os.path.join(flags.logdir, "validation"+tag, current_time)
    test_logdir = os.path.join(flags.logdir, "test"+tag, current_time)

    tf.reset_default_graph()
    X = dict()
    pred = dict()
    pred_scaled = dict()
    XH = dict()
    X = tf.placeholder(tf.float32, shape=[None, IMAGESIZE_Y, IMAGESIZE_X, 3], name="X0")
    y = tf.placeholder(tf.float32, shape=[None, IMAGESIZE_Y, IMAGESIZE_X, 1], name="y")
    mode = tf.placeholder(tf.bool, name="mode")
    trainStep = tf.placeholder(tf.float32, name="trainStep")
    netEpoch = tf.placeholder(tf.float32, name="epoch")
    tf.summary.scalar("trainStep",trainStep)
    tf.summary.scalar("epoch",netEpoch)

    print('Building network ..')

    unets = 1
    pred = make_unet(X, mode)


    variables=dict()

    StepDebug = tf.placeholder(tf.float32, name='StepDebug')
    tf.summary.scalar("StepDebug",StepDebug)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred)

    mask_dual = tf.squeeze(tf.stack((pred,pred, y), axis=3))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    IOU_op = realIOU(pred, y)
    tf.summary.scalar("IOU", IOU_op)
    tf.summary.scalar("realIOU", realIOU(pred, y))

    summary_op = tf.summary.merge_all()

    max_epoch = 0 

    for dirpath, dirnames, filenames in os.walk(flags.modeldir):
        for fname in filenames:
            fname,ext = os.path.splitext(fname)
            fname,ext = os.path.splitext(fname)
            if ('_' in fname):
                epoch = int(fname.split('_')[-1])
                if (epoch>max_epoch):
                    max_epoch=epoch
                    stem = '_'.join(fname.split('_')[:-1])

    print('Max epoch is: ',max_epoch,stem)

    with tf.Session() as sess:
        validation_summary_writer = tf.summary.FileWriter(val_logdir)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=100)

        try:
            global_step = tf.train.get_global_step(sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            stepCounter=0
            learnStep=0
            epoch=0
            for epoch in np.arange(max_epoch-19, max_epoch+1):
                learnStep = epoch % 7
                try:
                    checkpointfile = flags.modeldir+os.sep+'%s_%d.ckpt' % (stem, epoch)
                    saver.restore(sess, checkpointfile)
                except:
                    print('Unable to restore previous network in ', checkpointfile)
                    continue

                try:
                    print('Epoch %d, Queue Size of request queue: %d (empty: %d, learnStep: %d)' % (epoch,  requestQueue.qsize(),int(requestQueue.empty()), learnStep))
                except:
                    print('Queue emptyness: ', requestQueue.empty())

                # lengthy validation eval
                for k in range(int(5000/batchsize)):
                    requestQueue.put(LearningStep.VALIDATION)
                
                truePositives = falsePositives = trueNegatives = falseNegatives = 0
                intersection = 0
                union = 0
                idx=0
                while (requestQueue.qsize() > 0) or (getterQueue.qsize()>0):
                    (imgBatch,out,lmode) = getImageFromQueue(getterQueue, requestQueue, DB)
                
                    TFTimerID = TIMER.startTimer('TF')

                    [pred_img] = sess.run([pred],
                                feed_dict={X:imgBatch,
                                StepDebug: stepCounter,
                                netEpoch: epoch, 
                                y: out,
                                trainStep : learnStep+1,
                                mode: False})
                    idx+=1
                    TIMER.stopTimer('TF', TFTimerID)

                    im1 = out
                    im2 = pred_img

                    truePositives += np.sum(np.float32((im1>0.5) & (im2>0.5)))
                    falsePositives += np.sum(np.float32((im1<=0.5) & (im2>0.5)))
                    trueNegatives += np.sum(np.float32((im1<=0.5) & (im2<=0.5)))
                    falseNegatives += np.sum(np.float32((im1>0.5) & (im2<=0.5)))
                    # print('Max pred_img: ',np.max(pred_img),'GT:',np.max(out),'Mean pred_img: ',np.mean(pred_img),'GT:',np.mean(out),'STEP iou: ',step_iou)

                    intersection = truePositives
                    union = truePositives + falsePositives + falseNegatives


                PRC = truePositives/(truePositives+falsePositives+1E-4)
                REC = truePositives/(truePositives+falseNegatives+1E-4)
                iou = intersection/union
                F1 = 2* PRC * REC / (PRC+REC+1E-4)

                summary = tf.Summary()

                summary.value.add(tag='IOU', simple_value=(iou))
                summary.value.add(tag='F1', simple_value=(F1))
                summary.value.add(tag='Prec', simple_value=(PRC))
                summary.value.add(tag='Recall', simple_value=(REC))
                validation_summary_writer.add_summary(summary, global_step=epoch)

                statusline = 'epo-ch %d,  PRC = %.2f,  REC = %.2f, F1 = %.2f, mIOU = %.2f, TP: %d, TN: %d, FP: %d, FN: %d \n ' % (epoch, PRC,REC, 2 * PRC * REC / (PRC+REC), iou, truePositives, trueNegatives, falsePositives, falseNegatives)
                print(statusline)
                valfile = open('vallast10_%s.txt'% tag, mode='a')
                valfile.write(statusline)
                valfile.close()



            resfile = open('results_%s.txt'% tag, mode='w')

        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    flags = read_flags()
    main(flags)
