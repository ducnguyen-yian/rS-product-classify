import os
import logging
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
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
from util.file import tar_gz_decompress

from src.base_model_trainer import BaseModelTrainer
from src.resnet_model_trainer import ResnetHierarchicalModel
from src.metrics import *
from src.losses import *
from src.hierarchy import *
from src.class_imbalance import *
from src.data_generators import *

from sklearn.feature_extraction.text import CountVectorizer

from math import floor

class ConcatenatedModelTrainer(ResnetHierarchicalModel):
  def __init__(self):
    super().__init__()
  
  def get_arg_parser(self):
    parser = super().get_arg_parser()
    parser.add_argument('--text_cols', type=str, default='title,description')
    parser.add_argument('--combination_type', type=str, default='1')
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser
  
  def load_requirements(self):
    super().load_requirements()
    
    comb_types={'0': CombinationType.SUM, '1': CombinationType.CONCAT}
    comb_type=comb_types[self.args.combination_type]
    self.args.combination_type = comb_type
    
    self.args.text_cols = self.args.text_cols.split(",")
    self.vectorizer=vectorizer=CountVectorizer()
    
    self.training_dataframe=balance_classes(self.training_dataframe, self.args.max_count, self.vectorizer, self.H.num_leaves)
    self.write_training_dataframe('training_dataframe.json')
    
    df = self.training_dataframe
    text_cols = self.args.text_cols + []
    
    all_text=''
    
    while len(text_cols) > 0:
      all_text += ' ' + df[text_cols.pop()]

    self.vectorizer.fit(all_text)
    
    encoding = vectorizer.transform([df[self.args.text_cols[0]].iloc[0]]).toarray()
    self.text_model_input_size = encoding.shape[1]
    if comb_type is CombinationType.CONCAT:
      # if this is the case, we need vectorizer_size * num_text_cols
      self.text_model_input_size *= len(self.args.text_cols)
 
  def get_model(self):
    input_shape = self.get_input_shape(self.args.image_w, self.args.image_h)

    base_img_model = tf.keras.applications.ResNet50(
      include_top=False,
      weights='imagenet',
      input_tensor=tf.keras.layers.Input(shape=input_shape)
    )

    resnet_output = base_img_model.output
    resnet_pooled = tf.keras.layers.GlobalAveragePooling2D()(resnet_output)
    img_dense = tf.keras.layers.Dense(1024, 
                              activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(self.args.weight_decay),
                              bias_regularizer=tf.keras.regularizers.l2(self.args.weight_decay)
                             )(resnet_pooled)
    img_dropout = tf.keras.layers.Dropout(self.args.dropout)(img_dense)

    for idx, layer in enumerate(base_img_model.layers):
      if idx >= self.args.num_lock_layers:
        break
      layer.trainable = False
    
    text_input=tf.keras.layers.Input(shape=self.text_model_input_size)
    text_dense_1=tf.keras.layers.Dense(256, activation='relu')(text_input)
    text_dropout=tf.keras.layers.Dropout(self.args.dropout)(text_dense_1)
    text_dense_2=tf.keras.layers.Dense(256, activation='relu')(text_dropout)
    
    model_concat=tf.keras.layers.concatenate([img_dropout, text_dense_2], axis=-1)
    
    self.num_classes = self.H.num_leaves

    predictions=tf.keras.layers.Dense(self.num_classes, activation='softmax')(model_concat)

    model=tf.keras.models.Model(inputs=[base_img_model.input, text_input], outputs=predictions)
    
    return model
  
  def create_data_generators(self):
    dg = ImageTextDataGenerator(self.vectorizer, preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
      rotation_range=self.args.rotation_range,
      brightness_range=[1-self.args.brightness_range, 1+self.args.brightness_range],
      shear_range=self.args.shear_range,
      zoom_range=self.args.zoom_range,
      channel_shift_range=self.args.channel_shift_range,
      horizontal_flip=self.args.horizontal_flip
    )
    
    self.training_set = dg.flow_from_dataframe(
      self.training_dataframe, text_cols=self.args.text_cols, combination_type=self.args.combination_type, batch_size=self.args.batch_size, target_size=(256, 256)
    )
              
    self.validation_set = dg.flow_from_dataframe(
      self.validation_dataframe, text_cols=self.args.text_cols, combination_type=self.args.combination_type, batch_size=self.args.batch_size, target_size=(256, 256)
    )

    self.testing_set = dg.flow_from_dataframe(
      self.testing_dataframe, text_cols=self.args.text_cols, combination_type=self.args.combination_type, batch_size=self.args.batch_size, target_size=(256, 256), shuffle=False
    )
  
  def train(self):
    mc_filename='mdl_best_wts.hdf5'
    mc_filepath=os.path.join(self.args.sm_model_dir, mc_filename)
    
    category_loss = tf.keras.losses.CategoricalCrossentropy()
    
    hierarchy_metric = HierarchyMetric(self.H)
    hierarchy_loss = hierarchical_loss(self.H)
    
    fallback_metric=FallbackAccuracy(self.H, self.args.threshold)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)
    metrics = ["accuracy", hierarchy_metric, fallback_metric]
    
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
    [validation_loss, validation_accuracy, validation_hierarchy, validation_fallback, *_] = self.model.evaluate(self.validation_set) + [0]
    print("Validation-Loss: " + str(validation_loss) + ";")
    print("Validation-Accuracy: " + str(validation_accuracy) + ";")
    print("Validation-Hierarchy: " + str(validation_hierarchy) + ";")
    print('Validation-Fallback: ' + str(validation_fallback) + ';')

    print("Testing performance:")
    [test_loss, test_accuracy, test_hierarchy, test_fallback, *_] = self.model.evaluate(self.testing_set) + [0]
    print("Test-Loss: " + str(test_loss) + ";")
    print("Test-Accuracy: " + str(test_accuracy) + ";")
    print("Test-Hierarchy: " + str(test_hierarchy) + ";")
    print('Test-Fallback: ' +str(test_fallback) + ';')
    
    self.write_predictions('pred-post-finetune.npy')
    
if __name__ == "__main__":
  model = ConcatenatedModelTrainer()
  model.train()
  model.test()
  if model.is_training_box:
    model.save()  