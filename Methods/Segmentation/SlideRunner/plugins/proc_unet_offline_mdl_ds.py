import SlideRunner.general.SlideRunnerPlugin as SlideRunnerPlugin
from queue import Queue
import tensorflow as tf
import threading
import openslide
import numpy as np
import os
import cv2
from threading import Thread
import time


slidelist_1 = ["28_08_A_1_MCT Mitose 2017.svs", "2253_06_A_1_MCT Mitose 2017.svs", "2281_14_A_1_MCT Mitose 2017.svs", "2958_14_1A__1_MCT Mitose 2017.svs", "3806_09_B_1_MCT Mitose 2017.svs", "3786_09 A MCT Mitose 2017.svs", "5187_11 B MCT Mitose 2017.svs", "221_08 MCT Mitose 2017.svs", "1410_08_A_1_MCT Mitose 2017.svs", "1659_08_1_MCT Mitose 2017.svs"]
slidelist_2 = ["3541_11_A_1_MCT Mitose 2017.svs", "202_10_A_1_MCT Mitose 2017.svs", "6196_15_A_MCT Mitose 2017.svs", "4115_15 B MCT Mitose 2017.svs", "749_08_1_MCT Mitose 2017.svs", "1324_08_1_MCT Mitose 2017.svs", "1379_08_B_1_MCT Mitose 2017.svs", "127_10 1A MCT Mitose 2017.svs", "1359_08 A MCT Mitose 2017.svs", "4439_14_A_1_MCT Mitose 2017.svs"]
slidelist_3 = ["3679_09_B_1_MCT Mitose 2017.svs", "1075_09 H MCT Mitose 2017-001.svs", "1837_15 A MCT Mitose 2017.svs", "1911_08_1_MCT Mitose 2017.svs", "4792_13 A MCT Mitose 2017.svs", "4048_14_1_MCT Mitose 2017.svs", "1119_08_1_MCT Mitose 2017.svs", "2077_08_1_MCT Mitose 2017.svs", "296_09 B 1.2 8H 0.5E.svs", "406_14_1_MCT Mitose 2017.svs"]

Models_Extensions = [['.unet_bs6', 'models_unet_ijcars_bs6_1'],
                     ['.unet_bs3_take2', 'models_unet_ijcars_bs3_take2_1'],
                     ['.unet_bs3', 'models_unet_ijcars_bs3'],
                     ['.gtdata_final', 'GT'],
                     ['.unet_bs3_take3', 'models_unet_ijcars_bs3_take3_1' ],
                     ['.unet_bs3_take4', 'models_unet_ijcars_bs3_take4_1' ],
                     ['.unet_bs3_take5', 'models_unet_ijc_relme61_1' ],
                     ['.unet_bs6', 'models_unet_ijcars_bs6_take4_1' ],
                     ['.unet_bs6', 'models_unet_ijcars_bs6_take4_2' ],
                     ['.unet_bs6', 'models_unet_ijcars_bs6_take4_3' ],
                     ['.unet_tupac', 'models_unet_tupac_bs25_1'],
                     ['.unet_tupac', 'models_unet_tupac_bs25_2'],
                     ['.unet_tupac', 'models_unet_tupac_bs25_3'],
                     ['.unet_roi', 'models_unet_roi_1'],
                     ['.unet_roi', 'models_unet_roi_2'],
                     ['.unet_roi', 'models_unet_roi_3'],
                     ]



IMAGESIZE_X = 512
IMAGESIZE_Y = 512

margin = 64

TILESIZE_X = IMAGESIZE_X-2*margin
TILESIZE_Y = IMAGESIZE_Y-2*margin

PROC_ZOOM_LEVELS = [1, 2,4,8,16,32]

def get_np_array_from_tar_object(tar_extractfl):
     '''converts a buffer from a tar file in np.array'''
     return np.asarray(
        bytearray(tar_extractfl.read())
        , dtype=np.uint8)


class Plugin(SlideRunnerPlugin.SlideRunnerPlugin):
    version = 0.0
    shortName = 'Mitosis UNET offline (model select, downsampled)'
    inQueue = Queue()
    outQueue = Queue()
    description = 'Find mitotic figures using UNET'
    pluginType = SlideRunnerPlugin.PluginTypes.WHOLESLIDE_PLUGIN
    outputType = SlideRunnerPlugin.PluginOutputType.BINARY_MASK
    modelInitialized=False
    updateTimer=1.0
    slideFilename = None

    models=list()

    for [dirname, model] in Models_Extensions:
        models.append(model)

    configurationList = list((SlideRunnerPlugin.PluginConfigurationEntry(uid=0, name='Density Radius', initValue=0.00, minValue=0.0, maxValue=255.0),
                            SlideRunnerPlugin.PluginConfigurationEntry(uid=1, name='Amplification', initValue=4.00, minValue=1.0, maxValue=10.0),
                            SlideRunnerPlugin.ComboboxPluginConfigurationEntry(uid=2, name='Model', options=models),
                            SlideRunnerPlugin.PluginConfigurationEntry(uid=3, name='Epoch', initValue=134.00, minValue=130.0, maxValue=150.0),))

    def __init__(self, statusQueue:Queue):
        self.statusQueue = statusQueue
        self.preprocessQueue = Queue()
        self.preprocessorOutQueue = Queue()
        self.p = Thread(target=self.queueWorker, daemon=True)
        self.p.start()
        self.EXTENSION=''
        self.pp = dict()
        self.lastPlugin=-1
        for k in range(5):
            self.pp[k] = Thread(target=self.preprocessWorker, daemon=True)
            self.pp[k].start()
    
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


    def make_unet(self, Xhinted, training):
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
        conv1, pool1 = self.conv_conv_pool(net, [8, 8], training, name=1)
        conv2, pool2 = self.conv_conv_pool(pool1, [16, 16], training, name=2)
        conv3, pool3 = self.conv_conv_pool(pool2, [32, 32], training, name=3)
        conv4, pool4 = self.conv_conv_pool(pool3, [64, 64], training, name=4)
        conv5 = self.conv_conv_pool(pool4, [128, 128], training, name=5, pool=False)

        up6 = self.upsample_concat(conv5, conv4, name=6)
        conv6 = self.conv_conv_pool(up6, [64, 64], training, name=6, pool=False)

        up7 = self.upsample_concat(conv6, conv3, name=7)
        conv7 = self.conv_conv_pool(up7, [32, 32], training, name=7, pool=False)

        up8 = self.upsample_concat(conv7, conv2, name=8)
        conv8 = self.conv_conv_pool(up8, [16, 16], training, name=8, pool=False)

        up9 = self.upsample_concat(conv8, conv1, name=9)
        conv9 = self.conv_conv_pool(up9, [8, 8], training, name=9, pool=False)

        return tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=tf.nn.sigmoid, padding='same')
    

    def initializeModel(self):
        self.model = dict()


        tf.reset_default_graph()
        self.netEpoch = tf.placeholder(tf.float32, name="epoch")


        self.model['X'] = tf.placeholder(tf.float32, shape=[None, IMAGESIZE_Y, IMAGESIZE_X, 3], name="X0")
        self.model['y'] = tf.placeholder(tf.float32, shape=[None, IMAGESIZE_Y, IMAGESIZE_X, 1], name="y")
        self.model['mode'] = tf.placeholder(tf.bool, name="mode")
        self.trainStep = tf.placeholder(tf.float32, name="trainStep")

        self.model['pred'] = self.make_unet(self.model['X'], self.model['mode'])

        pluginDir = os.path.dirname(os.path.realpath(__file__))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
 

        summary_op = tf.summary.merge_all()

        self.sess = tf.Session()

        print('Initializing variables')
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print('Done.')
        self.currentModelCheckpoint=''

        self.saver = tf.train.Saver()

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
                            print('Unable to load',realfname,'in',modelpath)
                            exit()

        if not modelFound:
            print('Model not found in ',modelpath)
            self.setMessage('Model not found in'+str(modelpath))
            return False
        return True




    def queueWorker(self):
        self.targetModel = ''
        self.targetEpoch=-1
        self.lastCoordinates = ((None,None,None,None))
        modelInitialized=False
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

            model = Models_Extensions[job.configuration[2]][1]
            if not (model == self.targetModel) and (self.targetEpoch is not int(job.configuration[3])):
                if (self.loadModel(model, int(job.configuration[3]))):
                    self.targetModel=model
                    self.targetEpoch=int(job.configuration[3])
                else:
                    self.targetModel=''
                    self.targetEpoch=-1

            self.EXTENSION = Models_Extensions[job.configuration[2]][0]

            if (job.configuration[3]<150):
                self.EXTENSION += "_epoch%d" % int(job.configuration[3])

            if (self.slideFilename is None):
                print('Processing complete slide ..')
                self.processWholeSlide(job)
                self.slideFilename = job.slideFilename
    
                




    def preprocessWorker(self):
        while (True):
            if self.preprocessorOutQueue.qsize()<100:
                (tile_x, tile_y, sl, coordinates, tile_current) = self.preprocessQueue.get()

                try:
                    tn = sl.read_region(location=(int(coordinates[0]-margin), int(coordinates[1]-margin)),
                                level=0,size=(int(coordinates[2]-coordinates[0]+2*margin), int(coordinates[3]-coordinates[1]+2*margin)))
                    X_test = np.float32(cv2.cvtColor(np.array(tn), cv2.COLOR_BGRA2RGB))[:,:,::-1]
                    self.preprocessorOutQueue.put((X_test, tile_x, tile_y, coordinates, tile_current))
 
                except Exception as e:
                        self.preprocessQueue.put((tile_x, tile_y, sl, coordinates, tile_current))
                        print('Did not work, message is: ',e)
                        print('Tiles: ',tile_x,tile_y)
                        print('Coordinates:')
                        print('Slide dimensions:',sl.dimensions)
                        print('Coordinates are:',(int(coordinates[0]-margin), int(coordinates[1])), 
                                (int(coordinates[2]-coordinates[0]+2*margin), int(coordinates[3]-coordinates[1]+1*margin)))


            else:
                time.sleep(0.1)


    def worldCoordinatesToGridcoordinates(self,x,y) -> ((int, int), (int, int)):

        tile_indices = (int(np.floor(x/TILESIZE_X)),int(np.floor(y/TILESIZE_Y)))
        onTile_coordinates = (int(np.mod(x, TILESIZE_X)), int(np.mod(y, TILESIZE_Y)))

        return (tile_indices, onTile_coordinates)

    def gridCoordinatesToWorldCoordinates(self,tile_indices : (int,int), onTile_coordinates: (int,int)) -> (int,int):
        
        return (int(tile_indices[0]*TILESIZE_X+onTile_coordinates[0]),
                int(tile_indices[1]*TILESIZE_Y+onTile_coordinates[1]))


    def processWholeSlide(self, job):
        filename = job.slideFilename
        sl = openslide.open_slide(filename)
        ds=32
        self.imageMap = np.zeros((int(sl.dimensions[1]/ds),int(sl.dimensions[0]/ds)))
        self.slide = sl
        tiles_total_x = int(np.ceil(sl.dimensions[0] / TILESIZE_X))
        tiles_total_y = int(np.ceil(sl.dimensions[1] / TILESIZE_Y))

        tiles_total = tiles_total_x * tiles_total_y

        basefilename = job.slideFilename+self.EXTENSION+'_UNET.npz'

        if not os.path.exists(basefilename) and (self.targetModel is not ''):
            tiles2process_total = tile_current = 0
            for tile_y in range(tiles_total_y):
                for tile_x in range(tiles_total_x):
                    tile_current += 1


                    coords_p1 = self.gridCoordinatesToWorldCoordinates((tile_x,tile_y), (0,0))
                    coords_p2 = self.gridCoordinatesToWorldCoordinates((tile_x,tile_y), (TILESIZE_X,TILESIZE_Y))
    #                out_fullscale = self.processTile(sl, coordinates=(coords_p1+coords_p2))
                    coordinates = coords_p1+coords_p2
                    self.preprocessQueue.put((tile_x, tile_y, sl, coordinates, tile_current))
                    tiles2process_total+=1

            out_tile_current=0
            out_tile_num=0
            ds = 32
            print('Image size:', sl.dimensions)
            print('Images to be processed count: ', tiles2process_total)
            while (out_tile_num < tiles2process_total):
                out_tile_num+=1
                (X_test, tile_x, tile_y, coordinates, out_tile_current) = self.preprocessorOutQueue.get()
                out_fullscale = self.evaluateImage(X_test)
                target_size=(int(TILESIZE_Y/ds),int(TILESIZE_X/ds))
                out_ds = cv2.resize(out_fullscale, dsize=(target_size))
                target_x = int(coordinates[0]/ds)
                target_y = int(coordinates[1]/ds)
                #print('from coord: ',coordinates, target_x, target_y)
                if (coordinates[0]+IMAGESIZE_X<sl.dimensions[0]) and (coordinates[1]+IMAGESIZE_Y<sl.dimensions[1]):
                    self.imageMap[target_y:target_y+target_size[0], target_x:target_x+target_size[1]] = out_ds
                else:
                    remainsize = self.imageMap[target_y:target_y+target_size[0], target_x:target_x+target_size[1]].shape
                    self.imageMap[target_y:target_y+target_size[0], target_x:target_x+target_size[1]] = out_ds[0:remainsize[0],0:remainsize[1]]
                
                self.setProgressBar(100*out_tile_num/tiles_total)
            
            np.savez(basefilename, imageMap = self.imageMap)
        elif os.path.exists(basefilename):
            f = np.load(basefilename)
            self.imageMap = f['imageMap']
        else:
            self.setMessage('No cache and no model found.')
        
        x,y,w,h = job.coordinates
        retImg = cv2.getRectSubPix(np.float32(np.copy(self.imageMap)), (int(w/ds),int(h/ds)), center=((x-0.5*w)/ds,(y-0.5*h)/ds))
        retImg = cv2.resize(retImg, dsize=(job.currentImage.shape[1],job.currentImage.shape[0]) )
        self.returnImage(retImg, job.procId)

        self.setProgressBar(-1)
        

    def evaluateImage(self, image):
        import cv2
        
        overlap = 128
        margin = 64
        gridCoords_relative = list()
        # split into chunks

        X_test = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        X_test = np.reshape(X_test, newshape=[1,512,512,3])

        y_est = self.sess.run(
            [self.model['pred']],
            feed_dict={self.model['X']: X_test,
                        self.model['mode']: False})

        y_est = np.float32(np.asarray(y_est).squeeze())
        mapOut = y_est[margin:IMAGESIZE_Y-margin,margin:IMAGESIZE_X-margin]

        return mapOut
