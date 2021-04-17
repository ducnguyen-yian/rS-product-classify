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
from src.metrics import *
from src.losses import *
from src.hierarchy import *
from src.class_imbalance import *


class ResnetHierarchicalModel(BaseModelTrainer):
  
  def __init__(self):
    super().__init__()
  
  def get_arg_parser(self):
    parser = super().get_arg_parser()
    parser.add_argument('--num_finetune_epochs', type=int, default=1)
    parser.add_argument('--num_lock_layers', type=int, default=100)
    parser.add_argument('--predictions_s3_path', type=str, default="s3://data-science-product-image/training-output")
    # l2 regularization
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--max_count', type=int, default=1000)
    parser.add_argument('--rotation_range', type=int, default=5)
    parser.add_argument('--brightness_range', type=float, default=0.05)
    parser.add_argument('--shear_range', type=float, default=0.2)
    parser.add_argument('--zoom_range', type=float, default=0.05)
    parser.add_argument('--channel_shift_range', type=float, default=0.05)
    parser.add_argument('--horizontal_flip', type=bool, default=True)
    return parser
  
  def add_image_path(self, dataframe):
    # all images are in one spot, but the data frames will be split
    dataframe["image_path"] = dataframe["image"].apply(lambda image: os.path.join(self.args.train, "images", image))
  
  def load_requirements(self):
    # create a new subdirectory for prediction outputs on s3:
    self.args.predictions_s3_path += "/" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')    
    
    # load the taxonomy:
    with open(os.path.join(self.args.train, "taxonomy.json")) as taxonomy_file:
      self.taxonomy = json.load(taxonomy_file)
    
    # setup the hierarchy:
    self.H = Hierarchy(self.taxonomy, children_key="subcategories")
    
    # load the dataframes:
    self.training_dataframe = pd.read_json(os.path.join(self.args.train, "dataframes", "training.json"))
    self.testing_dataframe  = pd.read_json(os.path.join(self.args.train, "dataframes", "testing.json"))
    
    self.add_image_path(self.training_dataframe)
    self.add_image_path(self.testing_dataframe)
    
    n = len(self.training_dataframe)
    num_training = int(n * (1-self.args.validation_split))
    self.validation_dataframe = self.training_dataframe[num_training:]
    self.training_dataframe = self.training_dataframe[:num_training]
      
  def get_model(self):
    input_shape = self.get_input_shape(self.args.image_w, self.args.image_h)

    base_model = tf.keras.applications.ResNet50(
      include_top=False,
      weights='imagenet',
      input_tensor=tf.keras.layers.Input(shape=input_shape)
    )

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, 
                              activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(self.args.weight_decay),
                              bias_regularizer=tf.keras.regularizers.l2(self.args.weight_decay)
                             )(x)
    x = tf.keras.layers.Dropout(self.args.dropout)(x)
    
    self.num_classes = self.H.num_leaves

    predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    for idx, layer in enumerate(base_model.layers):
      if idx >= self.args.num_lock_layers:
        break
      layer.trainable = False
    
    model.summary()

    return model
  
  def create_data_generators(self):
    idg = tf.keras.preprocessing.image.ImageDataGenerator(
      preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
      rotation_range=self.args.rotation_range,
      brightness_range=[1-self.args.brightness_range, 1+self.args.brightness_range],
      shear_range=self.args.shear_range,
      zoom_range=self.args.zoom_range,
      channel_shift_range=self.args.channel_shift_range,
      horizontal_flip=self.args.horizontal_flip
    )
    
    y_col = [str(x) for x in list(range(0, self.num_classes))]
    
    self.training_set = idg.flow_from_dataframe(
      self.training_dataframe,
      x_col="image_path",
      y_col=y_col,
      class_mode="raw",
      target_size=(self.args.image_w, self.args.image_h),
      batch_size=self.args.batch_size
    )

    self.validation_set = idg.flow_from_dataframe(
      self.validation_dataframe,
      x_col="image_path",
      y_col=y_col,
      class_mode="raw",
      target_size=(self.args.image_w, self.args.image_h),
      batch_size=self.args.batch_size
    )

    self.testing_set = idg.flow_from_dataframe(
      self.testing_dataframe,
      x_col="image_path",
      y_col=y_col,
      class_mode="raw",
      target_size=(self.args.image_w, self.args.image_h),
      batch_size=self.args.batch_size, shuffle=False)
  
  def train(self):
    mc_filename='mdl_best_wts.hdf5'
    mc_filepath=os.path.join(self.args.sm_model_dir, mc_filename)
    
    category_loss = tf.keras.losses.CategoricalCrossentropy()
    
    hierarchy_metric = HierarchyMetric(self.H)
    hierarchy_loss = hierarchical_loss(self.H)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)
    metrics = ["accuracy", hierarchy_metric]
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=2, min_delta=0.02)
    model_checkpoint=tf.keras.callbacks.ModelCheckpoint(mc_filepath, save_best_only=True, monitor='val_accuracy', mode='max') 
    
    # first run it on categorical cross entropy
    self.model.compile(optimizer=optimizer, loss=category_loss, metrics=metrics)
    self.model.fit(self.training_set, validation_data=self.validation_set, epochs=self.args.num_epochs, callbacks=[early_stop, model_checkpoint])
    
    self.write_predictions('pred-pre-finetune.npy')
    
    # fine tune with hierarchical loss
    self.model.compile(optimizer=optimizer, loss=hierarchy_loss, metrics=metrics)
    self.model.fit(self.training_set, validation_data=self.validation_set, epochs=self.args.num_finetune_epochs, callbacks=[early_stop, model_checkpoint])
  
  def test(self):
    mc_filename='mdl_best_wts.hdf5'
    mc_filepath=os.path.join(self.args.sm_model_dir, mc_filename)
    
    self.model.load_weights(mc_filepath)
    
    print("Validation performance:")
    [validation_loss, validation_accuracy, validation_hierarchy, *_] = self.model.evaluate(self.validation_set) + [0]
    print("Validation-Loss: " + str(validation_loss) + ";")
    print("Validation-Accuracy: " + str(validation_accuracy) + ";")
    print("Validation-Hierarchy: " + str(validation_hierarchy) + ";")

    print("Testing performance:")
    [test_loss, test_accuracy, test_hierarchy, *_] = self.model.evaluate(self.testing_set) + [0]
    print("Test-Loss: " + str(test_loss) + ";")
    print("Test-Accuracy: " + str(test_accuracy) + ";")
    print("Test-Hierarchy: " + str(test_hierarchy) + ";")
    
    self.write_predictions('pred-post-finetune.npy')
    
  def save(self):
    self.model.save(os.path.join(self.args.sm_model_dir, '000000001'), save_format="tf")
  
  def write_predictions(self, pred_filename):
    self.testing_set.reset()
    probabilities = self.model.predict_generator(generator=self.testing_set)
    print("probabilities shape:", probabilities.shape)
    
    one_hot_true=[]
    
    for batch in self.testing_set:
      one_hot_true+=list(batch[1])
      if len(one_hot_true) >= len(probabilities):
        break
    one_hot_true=np.array(one_hot_true)
    print("one hot shape:", one_hot_true.shape)
    
    outputs = { "probabilities": probabilities, "one_hot_true": one_hot_true }
    
    if self.args.predictions_s3_path is not None:
      pred_filepath = os.path.join(self.args.sm_model_dir, pred_filename)
      np.save(pred_filepath, outputs)
      pred_s3_path = self.args.predictions_s3_path + "/" + pred_filename
      bucket, key = bucket_and_key_from_path(pred_s3_path)
      set_with_object_name(bucket, key, pred_filepath)

  def write_training_dataframe(self, training_df_filename):
    training_df_filepath = os.path.join(self.args.sm_model_dir, training_df_filename)
    self.training_dataframe.to_json(training_df_filepath)
    
    
if __name__ == "__main__":
  model = ResnetHierarchicalModel()
  model.train()
  model.test()
  if model.is_training_box:
    model.save()