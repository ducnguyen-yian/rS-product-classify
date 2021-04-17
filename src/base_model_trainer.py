import tensorflow as tf

import argparse
import os
import sys
import numpy as np
import pandas as pd
import json

dirpath = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dirpath, "..", "..", "data-science-core"))
from util.training import parse_args

class BaseModelTrainer:
  
  def __init__(self):
    self.args, _ = parse_args(self.get_arg_parser())
    self.load_requirements()
    self.model = self.get_model()
    self.create_data_generators()
  
  def get_arg_parser(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--image_w', type=int, default=256)
    parser.add_argument('--image_h', type=int, default=256)
    parser.add_argument('--validation_split', type=float, default=0.2)
    return parser
  
  def load_requirements(self):
    raise NotImplementedError()
    
  def get_model(self):
    raise NotImplementedError()
  
  def create_data_generators(self):
    raise NotImplementedError()
  
  @property
  def is_training_box(self):
    return self.args.hosts is not None and len(self.args.hosts) > 0 and self.args.current_host == self.args.hosts[0]
  
  def get_input_shape(self, w, h):
    if tf.keras.backend.image_data_format() == 'channels_first':
      input_shape = (3, w, h)
    else:
      input_shape = (w, h, 3)
    return input_shape
