# Training of a UNET-architecture for mitotic figure ROI detection

To train the model:
- python train_unet_roi.py <valrun>

To select a model stat:
- python modelSelection.py <valrun> <modeldir>
   (modeldir contains the model checkpoints, e.g. models_unet_roi_1)

For inference on the test data, please see my scripts Inference_xv<valrun>.py
    