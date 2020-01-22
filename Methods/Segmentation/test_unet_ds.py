import numpy as np

from SlideRunner.general.dependencies import *
from SlideRunner.plugins.proc_unet_offline_mdl_ds import Plugin
from queue import Queue


def test_plugins():
    print('Initializing model')
    statusQueue = Queue()
    plgn = Plugin(statusQueue)

    config = dict()
    config[0] = 1
    config[1] = 1

    print('Putting job into queue')

    models = np.array(plgn.configurationList[2].options)

    if (len(sys.argv)>2):
        slidename=sys.argv[1]
    else:
        print('Error: Missing arguments. Synopsis: test_unet_mdls.py slidename.svs model_name epoch')
        print('Models available: ',models)
        exit()

    modelidx = np.where(models==sys.argv[2])[0]
    if (len(modelidx)<1):
        print('Not found. Models are: ',models)
        exit()

    if (len(sys.argv)>3):
        epoch=int(sys.argv[3])
    else:
        epoch=50

    config[3] = epoch

    config[2] = modelidx[0]

    plgn.inQueue.put(SlideRunnerPlugin.jobToQueueTuple(currentImage=np.zeros((800,600,3)), slideFilename=slidename, coordinates=(21119, 22813, 1581, 1127), configuration=config))

    print('Waiting for progress')

    while (True):
        ret = plgn.statusQueue.get()
        status, msg = ret
        if (status == 0):
            print('New percentage: ', msg)
        if (status == 1):
            print(' -> ',msg)
        if (status == 0) and (msg == -1):
            sys.exit()
test_plugins()
