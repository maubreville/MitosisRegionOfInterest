import time
import os
import tensorflow as tf
import numpy as np
import sqlite3
import openslide
import cv2
import sys
import queue
import imageGetter_dirreg as imageGetter
import threading
from queue import Queue
import platform
from myTypes import *
from visualize import *
import matplotlib
import sklearn.metrics
matplotlib.use('Agg')

batchsize=12

basepath = '../../MITOS_WSI_CCMCT/WSI/'
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

slidelist_train = np.array(slidelist)[np.where([slidelist[x] not in slidelist_test for x in range(len(slidelist))])[0]].tolist()

PARALLEL_GETTERS = 6
IMAGESIZE_X = 384
IMAGESIZE_Y = 384

tag = 'reg_roi__'+sys.argv[1]

def get_filterKernel(dsamp, radius=25):
    filterKernel = np.zeros((63,63), dtype=np.float32)
    cv2.circle(filterKernel, center=(31,31), radius=25, color=[1], thickness=-1)
    filterKernel_small = cv2.resize(filterKernel, dsize=(round(63/dsamp),round(63/dsamp)))

    return filterKernel_small / np.sum(filterKernel_small)

def getFromQueue(queue, requestQueue):
        DB = sqlite3.connect(basepath+os.sep+'../databases/MITOS_WSI_CCMCT_ODAEL.sqlite')
        while (True):
            if queue.qsize()<100:
                lstep = requestQueue.get()

                im = np.zeros(shape=(batchsize,IMAGESIZE_Y,IMAGESIZE_X,3), dtype=np.uint8)
                reg = np.zeros(shape=(batchsize,1), dtype=np.float32)
                for k in range(batchsize):
                    r = np.random.rand(1)
                    if (lstep == LearningStep.TRAINING) or (lstep == LearningStep.VALIDATION):
                        if (r<0.33):                        
                            im[k], reg[k] = imageGetter.getImage(DB, slidelist_train, IMAGESIZE_X, IMAGESIZE_Y, learningStep=lstep,ds=16, basepath=basepath)
                        elif (r<0.66):
                            im[k], reg[k]= imageGetter.getImage(DB, slidelist_train, IMAGESIZE_X, IMAGESIZE_Y, learningStep=lstep,ds=16,basepath=basepath,allAreaPick=True)
                        else:
                            im[k], reg[k] = imageGetter.getImage(DB, slidelist_train, IMAGESIZE_X, IMAGESIZE_Y, learningStep=lstep,ds=16,basepath=basepath,allAreaPick=True)
                    else: # test
                        im[k], reg[k] = imageGetter.getImage(DB, slidelist_test, IMAGESIZE_X, IMAGESIZE_Y, learningStep=lstep,ds=16,basepath=basepath, allAreaPick=True)

#                    imgBatches_temp = np.append(np.append(im1,values=im2, axis=0), values=im3, axis=0)
#                    imgBatches = np.reshape(imgBatches, newshape=(3,512,512,3))
                
                queue.put((im,lstep,reg))
            else:
                time.sleep(0.1)




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
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.python.ops import array_ops

def focal_loss(sigmoid_p, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    #sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def my_focal_loss(y_true, y_pred_preactivation, alpha=0.25, gamma=2.0):
#  regression_batch = (Nx5), (x1,y1,x2,y2,state) where state = -1 ignore, 0 bg, 1 fg
#  labels_batch = (NX(C+1)), C = classes, last column = state

    with tf.variable_scope('FocalLoss'):
        labels         = y_true[:, :, :, :-1]
        anchor_state   = y_true[:, :, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred_preactivation

        # filter out "ignore" anchors
#        indices        = tf.where(tf.not_equal(anchor_state, -1))
#        indices           = tf.where(tf.equal(anchor_state, 1))


#        labels         = tf.gather_nd(labels, indices)
#        classification = tf.gather_nd(classification, indices)


        # compute the focal loss
        alpha_factor = tf.ones_like(labels) * alpha
        alpha_factor = tf.where(tf.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

#        cls_loss = focal_weight[:,0] * tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=classification)
        cls_loss = focal_weight * tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=classification)
    #    cls_loss = focal_weight * -1 * labels*tf.log(classification)

        # compute the normalizer: the number of positive anchors
#        normalizer = tf.where(tf.equal(anchor_state, 1))
#        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
#        normalizer = tf.maximum(1.0, normalizer)

        normalizer = tf.maximum(1.0, tf.reduce_sum(tf.cast(anchor_state, tf.float32)))


        return tf.reduce_sum(cls_loss) / normalizer


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

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
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

    wsl_out = tf.layers.conv2d(conv5_cla, 1, (1, 1), name='final', activation=None, padding='same', reuse=True)


    return tf.nn.sigmoid(tf.reshape(cla,[-1, 1])), tf.nn.sigmoid(wsl_out)


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

        intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) 
        tf.summary.scalar('Intersection',tf.reduce_mean(intersection))
        denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7
        tf.summary.scalar('denominator',tf.reduce_mean(denominator))

        return tf.reduce_mean(intersection / denominator) + 1e-7

def make_train_op(y_pred, y_true):
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
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
 
    global_step = tf.train.get_or_create_global_step()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

        optim = tf.train.AdamOptimizer(learning_rate=0.00001)

        return optim.minimize(loss, global_step=global_step), loss



def read_flags():
    """Returns flags"""

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('xval_num', type=int)

    parser.add_argument('--start_epoch',
                        default=0,
                        type=int)

    parser.add_argument("--epochs",
                        default=150,
                        type=int,
                        help="Number of epochs (default: 32)")

    parser.add_argument("--batch-size",
                        default=12,
                        type=int,
                        help="Batch size (default: 12)")

    parser.add_argument("--logdir",
                        default="logdir",
                        help="Tensorboard log directory (default: logdir)")

    parser.add_argument("--ckdir",
                        default="models_reg_roi_"+sys.argv[1],
                        help="Checkpoint directory (default: models)")

    parser.add_argument("--contdir",
                        default="",
                        help="Continue with old model state in directory")

    flags = parser.parse_args()
    return flags


def realIOU(y_pred, y_true):
    """Returns a IOU score (dice coefficient)

    Reference: 
        Rahman, M. A. and Wang, Y. (2016):
        Optimizing intersection-over-union in deep neural 
        networks for image segmentation. 
        In: International Symposium on Visual Computing, pages 234–244

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = intersection / ((y_pred + y_true - intersection).sum() + 1e-7)

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



    procs = list()
    for idx in range(PARALLEL_GETTERS):
        proc = threading.Thread(target=getFromQueue, args=(getterQueue,requestQueue,))
        proc.daemon=True
        proc.start()
        procs.append(proc)

    DB = sqlite3.connect(basepath+os.sep+'../databases/MITOS_WSI_CCMCT_ODAEL.sqlite')
    DBcur = DB.cursor()




    current_time = time.strftime("%m_%d_%H_%M_%S")
    train_logdir = os.path.join(flags.logdir,current_time, "train"+tag )
    val_logdir = os.path.join(flags.logdir,current_time, "validation"+tag)
    valtrain_logdir = os.path.join(flags.logdir,current_time, "validationTrain"+tag)
    test_logdir = os.path.join(flags.logdir,current_time, "test"+tag)

    tf.reset_default_graph()
    X = dict()
    pred = dict()
    pred_scaled = dict()
    XH = dict()
    subsf=16
    X = tf.placeholder(tf.float32, shape=[None, IMAGESIZE_Y, IMAGESIZE_X, 3], name="X0")
    y = tf.placeholder(tf.float32, shape=[None, 1], name="y")
    mode = tf.placeholder(tf.bool, name="mode")
    trainStep = tf.placeholder(tf.float32, name="trainStep")
    netEpoch = tf.placeholder(tf.float32, name="epoch")
    tf.summary.scalar("trainStep",trainStep)
    tf.summary.scalar("epoch",netEpoch)

    print('Building network ..')

    unets = 1
    pred, wsl_out = make_net(X, mode)

    model_vars = tf.trainable_variables()
    resnet_vars = [var for var in model_vars if 'resnet' in var.name]
    saverResnet = tf.train.Saver(resnet_vars)

    variables=dict()

    StepDebug = tf.placeholder(tf.float32, name='StepDebug')
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred)


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.variable_scope('Training'):
        with tf.control_dependencies(update_ops):
            train_ops=dict()
            train_op, loss = make_train_op(y_pred=pred,y_true=y )

    tf.summary.scalar('loss', loss)


    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_logdir)
        valtrain_summary_writer = tf.summary.FileWriter(valtrain_logdir)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=300)
        if (len(flags.contdir)>0) and os.path.exists(flags.contdir) and tf.train.checkpoint_exists(flags.contdir):
            latest_check_point = tf.train.latest_checkpoint(flags.contdir)
            saver.restore(sess, latest_check_point)
            print('Restored previous weights from ',flags.contdir)
        else:
            saverResnet.restore(sess, 'resnet_v1_50.ckpt')
            print('Restored RESNET50 weights.')

            try:
                os.rmdir(flags.ckdir)
            except:
                pass
            try:
                os.mkdir(flags.ckdir)
            except:
                pass

        try:
            global_step = tf.train.get_global_step(sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            stepCounter=0
            learnStep=0
            epoch=0
            for epoch in range(flags.epochs):
                learnStep = epoch % 7

                try:
                    print('Epoch %d, Queue Size of request queue: %d (empty: %d, learnStep: %d)' % (epoch,  requestQueue.qsize(),int(requestQueue.empty()), learnStep))
                except:
                    print('Queue emptyness: ', requestQueue.empty())

                if (epoch > 0):

                    for k in range(int(1000/batchsize)): # much smaller epochs
                        for j in range(10):                        
                            requestQueue.put(LearningStep.TRAINING)
                        requestQueue.put(LearningStep.VALIDATION)

                    try:
                        print('Filled -> Epoch %d, Queue Size of request queue: %d (empty: %d, learnStep: %d)' % (epoch, requestQueue.qsize(),  int(requestQueue.empty()), learnStep))
                    except:
                        print('Queue emptyness: ', requestQueue.empty())

                stepCounter=0
                valCounter=0
                while (requestQueue.qsize() > 0) or (getterQueue.qsize()>0):
                    (imgBatch,lmode,reg) = getterQueue.get()
                    stepCounter+=1

                    if (lmode == LearningStep.TRAINING):
                        _, step_mse, step_summary, global_step_value = sess.run(
                            [train_op, loss, summary_op, global_step],
                            feed_dict={X:imgBatch,
                                    StepDebug: stepCounter,
                                    netEpoch: epoch, 
                                    y: reg,
                                    trainStep : learnStep+1,
                                    mode: True})
                        if (global_step_value % 10 == 1):
                            train_summary_writer.add_summary(step_summary, global_step_value)
                    elif (lmode == LearningStep.VALIDATION):
                        valCounter+=1
                        tmode = (valCounter % 2 == 1)
                        step_mse, step_summary, global_step_value = sess.run(
                            [ loss, summary_op, global_step],
                            feed_dict={X:imgBatch,
                                    StepDebug: stepCounter,
                                    netEpoch: epoch, 
                                    y: reg,
                                    trainStep : learnStep+1,
                                    mode: tmode})
                        if (tmode):
                            valtrain_summary_writer.add_summary(step_summary, global_step_value)
                        else:
                            val_summary_writer.add_summary(step_summary, global_step_value)
                    # print('CLA loss: ',step_mse)
                # lengthy validation eval
                for k in range(50):
                    requestQueue.put(LearningStep.VALIDATION)
                
                truePositives = falsePositives = trueNegatives = falseNegatives = 0
                iouList = list()
                intersection = 0
                union = 0
                poserrors = 0
                negerrors = 0
                detarray = np.empty(0)
                truearray = np.empty(0)
                idx=0
                errors=0
                with open(train_logdir+os.sep+'epoch%d.html' % epoch,'w') as v_html:
                    v_html.write('<table><tr><td>Image</td><td>GT count</td><td>Pred Cnt</td><td>WSL Image</td></tr>')
                    while (requestQueue.qsize() > 0) or (getterQueue.qsize()>0):
                        (imgBatch,lmode,reg) = getterQueue.get()
                        step_loss, wsl_img, step_summary, global_step_value,pred_val = sess.run(
                            [ loss, wsl_out, summary_op, global_step, pred],
                            feed_dict={X:imgBatch,
                                    StepDebug: stepCounter,
                                    netEpoch: epoch, 
                                    y: reg,
                                    trainStep : learnStep+1,
                                    mode: False})
                        errors += np.mean(step_loss)
                        signed_error = pred_val - reg
                        poserrors += np.sum(np.square(signed_error[signed_error>0.0]))
                        negerrors += np.sum(np.square(signed_error[signed_error<=0.0]))
                        idx+=1
                        if (idx<10):
                            v_html.write('<tr><td><img src="'+'im%d_%d.png' % (epoch,idx)+'"></td><td>"'+str(reg)+'</td><td>'+str(pred_val)+'</td><td><img src="wsl_%d_%d.svg">' % (epoch,idx) + '</td></tr>')
                            cv2.imwrite(train_logdir+os.sep+'im%d_%d.png' % (epoch,idx), imgBatch[0])

                            import matplotlib.pyplot as plt

                            plt.figure()
                            plt.imshow(wsl_img[0,:,:,0])
                            plt.colorbar()
                            plt.tight_layout()
                            plt.savefig(train_logdir+os.sep+'wsl_%d_%d.svg' % (epoch,idx))


                    v_html.write('</table>')

                    summary = tf.Summary()

                    summary.value.add(tag='RMSpE', simple_value=np.sqrt(poserrors))
                    summary.value.add(tag='RMSnE', simple_value=np.sqrt(negerrors))
                    summary.value.add(tag='RMSE', simple_value=np.sqrt(np.mean(np.square(np.array(errors)))))
                    val_summary_writer.add_summary(summary, global_step=epoch)


                for k in range(50):
                    requestQueue.put(LearningStep.VALIDATION)
                
                truePositives = falsePositives = trueNegatives = falseNegatives = 0
                iouList = list()
                intersection = 0
                union = 0
                detarray = np.empty(0)
                truearray = np.empty(0)
                idx=0
                errors=0
                poserrors=0
                negerrors=0


                with open(valtrain_logdir+os.sep+'epoch%d.html' % epoch,'w') as v_html:
                    v_html.write('<table><tr><td>Image</td><td>GT count</td><td>Pred Cnt</td><td>WSL Image</td></tr>')
                    while (requestQueue.qsize() > 0) or (getterQueue.qsize()>0):
                        (imgBatch,lmode,reg) = getterQueue.get()
                        step_loss, wsl_img, step_summary, global_step_value,pred_val = sess.run(
                            [ loss, wsl_out, summary_op, global_step, pred],
                            feed_dict={X:imgBatch,
                                    StepDebug: stepCounter,
                                    netEpoch: epoch, 
                                    y: reg,
                                    trainStep : learnStep+1,
                                    mode: True})
                        errors += np.mean(step_loss)
                        signed_error = pred_val - reg
                        poserrors += np.sum(np.square(signed_error[signed_error>0]))
                        negerrors += np.sum(np.square(signed_error[signed_error<=0]))
                        idx+=1
                        if (idx<10):
                            v_html.write('<tr><td><img src="'+'im%d_%d.png' % (epoch,idx)+'"></td><td>"'+str(reg)+'</td><td>'+str(pred_val)+'</td><td><img src="wsl_%d_%d.svg">' % (epoch,idx) + '</td></tr>')
                            cv2.imwrite(valtrain_logdir+os.sep+'im%d_%d.png' % (epoch,idx), imgBatch[0])

                            import matplotlib.pyplot as plt

                            plt.figure()
                            plt.imshow(wsl_img[0,:,:,0])
                            plt.colorbar()
                            plt.tight_layout()
                            plt.savefig(valtrain_logdir+os.sep+'wsl_%d_%d.svg' % (epoch,idx))


                    v_html.write('</table>')

                    summary = tf.Summary()

                    summary.value.add(tag='RMSE', simple_value=np.sqrt(np.mean(np.square(np.array(errors)))))
                    summary.value.add(tag='RMSpE', simple_value=np.sqrt(poserrors))
                    summary.value.add(tag='RMSnE', simple_value=np.sqrt(negerrors))
                    valtrain_summary_writer.add_summary(summary, global_step=epoch)



#                if (epoch%25==0):
                if (epoch>=flags.epochs-20):
                    saver.save(sess, "%s/model_%d.ckpt" % (flags.ckdir, epoch))
                saver.save(sess, "{}/model.ckpt".format(flags.ckdir))


            # final evlauation on test set
            for k in range(400):
                requestQueue.put(LearningStep.TEST)
            

            if not os.path.exists(test_logdir):
                os.mkdir(test_logdir)

            with open(test_logdir+os.sep+'epoch%d.html' % epoch,'w') as v_html:
                v_html.write('<table><tr><td>Image</td><td>GT count</td><td>Pred Cnt</td><td>WSL Image</td></tr>')
                while (requestQueue.qsize() > 0) or (getterQueue.qsize()>0):
                    (imgBatch,lmode,reg) = getterQueue.get()
                
                    step_loss, wsl_img, step_summary, global_step_value,pred_val = sess.run(
                        [ loss, wsl_out, summary_op, global_step, pred],
                        feed_dict={X:imgBatch,
                                StepDebug: stepCounter,
                                netEpoch: epoch, 
                                y: reg,
                                trainStep : learnStep+1,
                                mode: False})

                    if (idx<10):
                        v_html.write('<tr><td><img src="'+'im%d_%d.png' % (epoch,idx)+'"></td><td>"'+str(reg)+'</td><td>'+str(pred_val)+'</td><td><img src="wsl_%d_%d.svg">' % (epoch,idx) + '</td></tr>')
                        cv2.imwrite(test_logdir+os.sep+'im%d_%d.png' % (epoch,idx), imgBatch[0])

                        import matplotlib.pyplot as plt

                        plt.figure()
                        plt.imshow(wsl_img[0,:,:,0])
                        plt.colorbar()
                        plt.tight_layout()
                        plt.savefig(test_logdir+os.sep+'wsl_%d_%d.svg' % (epoch,idx))



        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model.ckpt".format(flags.ckdir))


if __name__ == '__main__':
    flags = read_flags()
    main(flags)
