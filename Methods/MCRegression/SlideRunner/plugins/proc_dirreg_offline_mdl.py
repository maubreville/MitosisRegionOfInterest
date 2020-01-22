import SlideRunner.general.SlideRunnerPlugin as SlideRunnerPlugin
from queue import Queue
import tensorflow as tf
import threading
import openslide
import numpy as np
import os
import cv2
from threading import Thread
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.python.ops import array_ops
slim = tf.contrib.slim
import time
IMAGESIZE_X = 512
IMAGESIZE_Y = 512

Models_Extensions = [['dirreg', 'models_reg_roi_1'],
                     ['dirreg', 'models_reg_roi_2'],
                     ['dirreg', 'models_reg_roi_3'],
                    ]


margin = 0

TILESIZE_X = IMAGESIZE_X-2*margin
TILESIZE_Y = IMAGESIZE_Y-2*margin

PROC_ZOOM_LEVELS = [1, 2,4,8,16,32]

def conv_conv_pool( input_, n_filters, training, name, pool=True, activation=tf.nn.relu):
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
        #        net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
                net = activation(net, name="relu{}_{}".format(name, i + 1))

            if pool is False:
                return net

            pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

            return net, pool


    

def nms(coords_active, scores_active, threshold=0.3, doNms=True ):
    # circular nms for cell detection

    # similar to yolo, but using the radius as critereon

    # Suppose we have 2 circles with distance x, the IOU will only be dependent on the
    # distance and not on the actual coordinates. So we can simplify the problem to
    # a situation where the circles of equal radius are on the y-axis only, 
    # at positions 0 and +d

    # IOU threshold of 0.3 is set, 

    # circle1: (x)^2 + (y)^2 < r
    # circle2: (x-d)^2 + (y)^2 < r


    # asof: http://mathworld.wolfram.com/Circle-CircleIntersection.html
    # intersection = 2*r^2*acos(0.5*d/r)-0.5 ((-d+2r)  * (d) )
    #              = 0.5 d^2 - rd + 2*r^2 *acos(0.5*d/r)
    #              

#def intersection(d,r):
#      return 0.5* d**2 - r*d + 2*r**2 *np.arccos(0.5*d/r)

#def union(d,r):
###     return 2*np.pi*r**2 - intersection(d,r)
#def iou(d,r):
#     return intersection(d,r) / union(d,r)

    # For d=28.23 we achieve an IOU of 0.3 for r=25, like used in the retinanet approach

#    radius_thres = 28.23
    radius_thres = 30
    order = np.argsort(-scores_active)

    scores_active = list(scores_active[order])
    coords_active = list(coords_active[order])

    scores_maxima = list()
    coords_maxima = list()

    if (doNms):
        # Now do nonmaximum suppression
        while (len(scores_active)>0):
            scores_maxima.append(scores_active[0])
            coords_maxima.append(coords_active[0])

            j=1
            while (j<len(scores_active)):
                if np.sqrt(np.sum(np.square(coords_active[j]-coords_active[0])))<radius_thres:
                    del coords_active[j]
                    del scores_active[j]
                else:
                    j=j+1
            del coords_active[0]
            del scores_active[0]

        scores_active = np.asarray(scores_maxima)
        coords_active = np.asarray(coords_maxima)
    return coords_active, scores_active

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def ProcessDetection(reg_pred_list, cla_pred_list, threshold = 0.3, do_nms=True, batchId=0, target:tuple=(0,0)) -> (np.array, np.array):
    """
        Visualize detection results of Mitosis Detection approach
            reg_pred: Regression prediction (x,y)

            cla_pred: Classification (x,y,C==1)

    """

    allPositives = list()
    if not isinstance(cla_pred_list, (list,)): # pyramid is default
        cla_pred_list = list((cla_pred_list,))
        reg_pred_list = list((reg_pred_list,))
    
    scores_active = np.zeros((0,))
    coords_active = np.zeros((0,2))
    for pyramidLayer in range(len(cla_pred_list)):

        if (cla_pred_list[pyramidLayer].shape[-1]>1):
            cla_pred = cla_pred_list[pyramidLayer][batchId,:,:,1] 
        else:
            cla_pred = cla_pred_list[pyramidLayer][batchId,:,:,0] 
        reg_pred = reg_pred_list[pyramidLayer][batchId]

        if (len(cla_pred.shape)==4):
            # strip dimension
            cla_pred = np.resize(cla_pred, [cla_pred.shape[1], cla_pred.shape[1], 1])
            reg_pred = np.resize(reg_pred, [reg_pred.shape[1], reg_pred.shape[1], 2])

        normRadius=25
        ds = 16#int(img.shape[0] / reg_true.shape[0])

        anchor_x,anchor_y = np.meshgrid(ds*np.arange(reg_pred.shape[0]),ds*np.arange(reg_pred.shape[1]))

        coord_pred = np.zeros((reg_pred.shape[0], reg_pred.shape[1], 2))

        coord_pred[:,:,0] = -reg_pred[:,:,0]*normRadius+anchor_x+target[0]
        coord_pred[:,:,1] = -reg_pred[:,:,1]*normRadius+anchor_y+target[1]

        active_indices = np.where(cla_pred>threshold)


        scores_active = np.append(scores_active,cla_pred[active_indices])
        coords_active = np.vstack((coords_active, coord_pred[active_indices[0], active_indices[1],:]))



    coords_active, scores_active = nms(coords_active, scores_active, doNms=do_nms)


    return coords_active, scores_active



def make_net(Xhinted, training):
    """Build a ResNet FCN architecture

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

    with slim.arg_scope(resnet_v1.resnet_arg_scope(batch_norm_decay=0.9)):
            net, end_points = resnet_v1.resnet_v1_50(net,
                                                None,
                                                is_training=training,
                                                global_pool=False,
                                                reuse=False,
                                                output_stride=16)

#    net = tf.layers.conv2d(net, 4, (1, 1), name="color_space_adjust")
#    conv1, pool1 = conv_conv_pool(net, [8, 8], training, name=1)
#    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, name=2)
#    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, name=3)
#    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, name=4)

    conv5_cla = conv_conv_pool(net, [128, 128], training, name=5, pool=False)
    # N x X x y x 128

    mp = tf.reduce_mean(conv5_cla, axis=[1,2], keepdims=True)
    print('Output of GAP: ',mp)

    cla = tf.layers.conv2d(mp, 1, (1, 1), name='final', activation=None, padding='same')


    return tf.nn.sigmoid(tf.reshape(cla,[-1, 1]))


def get_np_array_from_tar_object(tar_extractfl):
     '''converts a buffer from a tar file in np.array'''
     return np.asarray(
        bytearray(tar_extractfl.read())
        , dtype=np.uint8)



class Plugin(SlideRunnerPlugin.SlideRunnerPlugin):
    version = 0.0
    shortName = 'Mitosis Detection CMT'
    inQueue = Queue()
    outQueue = Queue()
    description = 'Direct Regression of Mitotic Activity'
    pluginType = SlideRunnerPlugin.PluginTypes.WHOLESLIDE_PLUGIN
    outputType = SlideRunnerPlugin.PluginOutputType.BINARY_MASK
    modelInitialized=False
    updateTimer=0.1
    slideFilename = None

    models=list()

    for [dirname, model] in Models_Extensions:
        models.append(model)

    configurationList = list((SlideRunnerPlugin.PluginConfigurationEntry(uid=0, name='Detection Threshold', initValue=0.3, minValue=0.3, maxValue=1.0),
                              SlideRunnerPlugin.ComboboxPluginConfigurationEntry(uid=1, name='Model', options=models),
                              SlideRunnerPlugin.PluginConfigurationEntry(uid=2, name='Epoch', initValue=150.00, minValue=1.0, maxValue=150.0)))

    def __init__(self, statusQueue:Queue):
        self.statusQueue = statusQueue
        self.preprocessQueue = Queue()
        self.preprocessorOutQueue = Queue()
        self.slide = None
        self.p = Thread(target=self.queueWorker, daemon=True)
        self.p.start()
        self.currentModelCheckpoint = -1
        self.pp = dict()
        for k in range(10):
            self.pp[k] = Thread(target=self.preprocessWorker, daemon=True)
            self.pp[k].start()
    



    def worldCoordinatesToGridcoordinates(self,x,y) -> ((int, int), (int, int)):

        tile_indices = (int(np.floor(x/TILESIZE_X)),int(np.floor(y/TILESIZE_Y)))
        onTile_coordinates = (int(np.mod(x, TILESIZE_X)), int(np.mod(y, TILESIZE_Y)))

        return (tile_indices, onTile_coordinates)

    def gridCoordinatesToWorldCoordinates(self,tile_indices : (int,int), onTile_coordinates: (int,int)) -> (int,int):
        
        return (int(tile_indices[0]*TILESIZE_X+onTile_coordinates[0]),
                int(tile_indices[1]*TILESIZE_Y+onTile_coordinates[1]))

    def conv_conv_pool(self, input_, n_filters, training, name, pool=True, activation=tf.nn.relu):
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
                net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
                net = activation(net, name="relu{}_{}".format(name, i + 1))

            if pool is False:
                return net

            pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

            return net, pool


    def upsample_concat(self, inputA, input_B, name):
        """Upsample `inputA` and concat with `input_B`

        Args:
            input_A (4-D Tensor): (N, H, W, C)
            input_B (4-D Tensor): (N, 2*H, 2*H, C2)
            name (str): name of the concat operation

        Returns:
            output (4-D Tensor): (N, 2*H, 2*W, C + C2)
        """
        upsample = self.upsampling_2D(inputA, size=(2, 2), name=name)

        return tf.concat([upsample, input_B], axis=-1, name="concat_{}".format(name))


    def upsampling_2D(self,tensor, name, size=(2, 2)):
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

    def loadModel(self, modelpath, epoch):
        pluginDir = os.path.dirname(os.path.realpath(__file__))
        modelpath = pluginDir + os.sep + modelpath
        modelFound=False
        for dirpath,dirnames,files in os.walk(modelpath):
            for fname in files:
                realfname,ext = os.path.splitext(fname)
                if ('_%d.ckpt' % epoch) in realfname:
                    if (modelpath + os.sep + realfname is not self.currentModelCheckpoint):
                        try:
                            self.saver.restore(self.sess, modelpath + os.sep + realfname)
                            print('Restored ',realfname)
                            modelFound=True
                            self.currentModelCheckpoint = modelpath + os.sep + realfname
                        except:
                            print('Unable to load',realfname)
                            exit()
        if not modelFound:
            print('Model not found in ',modelpath)
            exit() 

    def initializeModel(self):
        self.model = dict()

        self.setMessage('DirReg init.')
        self.setProgressBar(0)

        tf.reset_default_graph()
        self.netEpoch = tf.placeholder(tf.float32, name="epoch")


        self.model['X'] = tf.placeholder(tf.float32, shape=[None, IMAGESIZE_Y, IMAGESIZE_X, 3], name="X0")
        self.model['y'] = tf.placeholder(tf.float32, shape=[None, IMAGESIZE_Y, IMAGESIZE_X, 1], name="y")
        self.model['mode'] = tf.placeholder(tf.bool, name="mode")
        self.trainStep = tf.placeholder(tf.float32, name="trainStep")

        self.model['pred'] = make_net(self.model['X'], self.model['mode'])


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        summary_op = tf.summary.merge_all()

        self.sess = tf.Session()

        print('Initializing variables')
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print('Done.')

        self.saver = tf.train.Saver()



    def queueWorker(self):
        self.targetModel=-1
        self.lastCoordinates = ((None,None,None,None))
        modelInitialized=False

        oldThres = 0.0

        print('Queue worker running.')
        quitSignal=False
        while not quitSignal:
            job = SlideRunnerPlugin.pluginJob(self.inQueue.get())

            if not (modelInitialized):
                self.initializeModel()
                modelInitialized=True

            if (job.jobDescription == SlideRunnerPlugin.JobDescription.QUIT_PLUGIN_THREAD):
                # signal to exit this thread
                quitSignal=True
                continue

            print('Received request for: ',job.slideFilename, job.coordinates, job.currentImage.shape)

            updateAnnos=False

            model = Models_Extensions[job.configuration[1]][1]
            if not (model == self.targetModel):
                self.targetModel=model
                self.loadModel(self.targetModel, int(job.configuration[2]))

            self.resultsStoreFilename = job.slideFilename + '_results_%s_%d.npz' % (Models_Extensions[job.configuration[1]][0],job.configuration[2])

            if (self.slideFilename is None):
                print('Processing complete slide ..')
                self.processWholeSlide(job)
                self.slideFilename = job.slideFilename

                updateAnnos=True
            
            if not (oldThres == job.configuration[0]):
                updateAnnos=True
                oldThres=job.configuration[0]





    def preprocessWorker(self):
        while (True):
            if self.preprocessorOutQueue.qsize()<500:
                (tile_x, tile_y, filename, coordinates, tile_current) = self.preprocessQueue.get()
                sl = openslide.open_slide(filename)

                tn = sl.read_region(location=(int(coordinates[0]-margin), int(coordinates[1]-margin)),
                        level=0,size=(int(coordinates[2]-coordinates[0]+2*margin), int(coordinates[3]-coordinates[1]+2*margin)))

                X_test = np.float32(cv2.cvtColor(np.array(tn), cv2.COLOR_BGRA2RGB))[:,:,::-1]
                X_test = cv2.cvtColor(X_test, cv2.COLOR_BGR2RGB)
                #X_test = np.reshape(X_test, newshape=[1,512,512,3])

                self.preprocessorOutQueue.put((X_test, tile_x, tile_y, coordinates, tile_current))
            else:
                time.sleep(0.1)



    def processWholeSlide(self, job):
        self.setMessage('DirReg preparing ..')
        self.setProgressBar(0)


        filename = job.slideFilename
        sl = openslide.open_slide(filename)

        if os.path.isfile(self.resultsStoreFilename):
            resultsStore = np.load(self.resultsStoreFilename)
            self.tilesProcessed = resultsStore['tilesProcessed'].tolist()
            self.scores = resultsStore['scores'].tolist()
        else:
            self.tilesProcessed = list()
            self.scores = list()

        self.slide = sl
        tiles_total_x = int(np.floor(sl.dimensions[0] / TILESIZE_X))
        tiles_total_y = int(np.floor(sl.dimensions[1] / TILESIZE_Y))

        tiles_total = tiles_total_x * tiles_total_y

        

        tiles2process_total = tile_current = 0
        for tile_y in range(tiles_total_y):
            for tile_x in range(tiles_total_x):
                tile_current += 1


                if ([tile_x, tile_y] in self.tilesProcessed):
                    continue

                coords_p1 = self.gridCoordinatesToWorldCoordinates((tile_x,tile_y), (0,0))
                coords_p2 = self.gridCoordinatesToWorldCoordinates((tile_x,tile_y), (TILESIZE_X,TILESIZE_Y))
#                out_fullscale = self.processTile(sl, coordinates=(coords_p1+coords_p2))
                coordinates = coords_p1+coords_p2
                self.preprocessQueue.put((tile_x, tile_y, filename, coordinates, tile_current))
                tiles2process_total+=1

        out_tile_current=0
        batchsize=20
        batchcounter=0
        batch = np.zeros((batchsize, IMAGESIZE_Y, IMAGESIZE_X, 3))
        batchTiles = list()

        print('Images to be processed count: ', tiles2process_total)
        import time

        gt = 0
        gtf = 0
        out_tile_count=0

        while (out_tile_count < tiles2process_total):
            t = time.time()
            (X_test, tile_x, tile_y, coordinates, out_tile_current) = self.preprocessorOutQueue.get()
            gt += time.time()-t

            batch[batchcounter] = X_test
            batchTiles.append([tile_x,tile_y])
            batchcounter+=1
            out_tile_count+=1

            if (batchcounter==batchsize) or (out_tile_count==tiles2process_total):
                gtf -= time.time()
                out = self.evaluateImage(batch[0:batchcounter])
                gtf += time.time()
#                print('TensorFlow: %.2f s, Preproc: %.2f s, QSize: %d ' % (gtf, gt, self.preprocessorOutQueue.qsize()))
                for k in range(out.shape[0]):
                    self.scores.append(out[k])
                self.tilesProcessed += batchTiles
                print('Length of out:' ,len(self.scores), len(self.tilesProcessed))
                if (len(self.scores) != len(self.tilesProcessed)):
                    print('Wrong length!')
                    sys.exit()
                batchTiles = list()
                batchcounter=0

                self.setProgressBar(100*out_tile_count/tiles_total)


            # split into chunks
            if (self.inQueue.qsize()>0): # new request pending
                self.saveState()
                return

        print('Length of out:' ,len(self.scores), len(self.tilesProcessed),out_tile_count)


        self.saveState()
        self.setProgressBar(-1)
        self.setMessage('DirReg calculation done.')


    def saveState(self):
        np.savez_compressed(self.resultsStoreFilename, scores=self.scores, tilesProcessed=self.tilesProcessed)

    def exceptionHandlerOnExit(self):
        if (self.slide):
            self.saveState()

    def getAnnotations(self):
        return self.annos

    def evaluateImage(self, image):
#        if (np.all(np.mean(image, axis=(1,2,3))<200)):
            [y_act]  = self.sess.run(
                [self.model['pred']],
                feed_dict={self.model['X']: image,
                            self.model['mode']: False})

            
            return y_act
#        else:
#            return np.zeros(image.shape[0])


