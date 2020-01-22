In this folder the scripts used to train the MC regression approach based on RESNET-50 can be found.

You need to have TensorFlow 1.x to run this code.

Training: 
- python3 train_regression.py <valrun>

Model selection:
- python3 modelselecection.py <valrun>
    
Inference:
- python3 <slidename>.svs <model_name> <epoch>

