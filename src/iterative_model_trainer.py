import os
import logging
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from tensorflow.python.keras import backend as K
import tensorflow as tf

import argparse
import sys
import numpy as np
import pandas as pd
import json
import datetime

from sklearn.metrics import confusion_matrix

dirpath = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dirpath, ".."))
sys.path.insert(1, os.path.join(dirpath, "..", "..", "data-science-core"))
from util.s3 import *
from src.base_model_trainer import BaseModelTrainer
from src.resnet_model_trainer import ResnetHierarchicalModel
from src.metrics import *
from src.losses import *
from src.hierarchy import *
from src.class_imbalance import *

class IterativeModelTrainer(ResnetHierarchicalModel):

  def get_arg_parser(self):
    parser = super().get_arg_parser()
    parser.add_argument('--model_path', type=str, default='home/ec2-user/SageMaker/data-science-product-image/files/model-files/image-model-2020-12-04-71pct/export/Servo/000000001')
    return parser

  def get_model(self):
    model = tf.keras.models.load_model(self.args.model_path)

    for idx, layer in enumerate(model.layers):
      if idx >= self.args.num_lock_layers:
        break
      layer.trainable = False
    
    model.summary()

    return model  

if __name__ == "__main__":
  model = IterativeModelTrainer()
  model.train()
  model.test()
  if model.is_training_box:
    model.save()  