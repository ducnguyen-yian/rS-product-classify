import tensorflow as tf
import numpy as np
from keras.preprocessing import image as krs_image
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
import types
from enum import Enum


class CombinationType(Enum):
  SUM = 0
  CONCAT = 1

def batch_generation_override(self, index_array):
  images, text, labels = ([], [], [])
  for i, j in enumerate(index_array):

    example = self.dataframe.iloc[j]

    img = load_img(example[self.image_col],
                   color_mode=self.color_mode,
                   target_size=self.target_size,
                   interpolation=self.interpolation)
    x = img_to_array(img, data_format=self.data_format)
    if hasattr(img, 'close'):
      img.close()
    if self.image_data_generator:
      params = self.image_data_generator.get_random_transform(x.shape)
      x = self.image_data_generator.apply_transform(x, params)
      x = self.image_data_generator.standardize(x)
    images.append(x)

    label = example[self.label_col]
    labels.append(label)
    
    embedding = self.vectorize([example[col] for col in self.text_cols])
    text.append(embedding)

  return [np.array(images), np.array(text)], np.array(labels)

class ImageTextDataGenerator(ImageDataGenerator):
  def __init__(self, vectorizer, **kwargs):
    self.vectorizer = vectorizer
    super().__init__(**kwargs)
  
  def vectorize(self, text):
    sparse = self.vectorizer.transform([text])
    return np.array(sparse.todense())[0]
  
  def sum_text(self, texts):
    reduced = ' '.join(texts)
    return self.vectorize(reduced)

  def concat_text(self, texts):
    vectorized = [self.vectorize(text) for text in texts]
    return np.concatenate(vectorized, axis=None)
  
  def flow_from_dataframe(self, 
                          dataframe, 
                          combination_type=CombinationType.SUM, 
                          image_col="image_path", 
                          text_cols=["title_and_description"], 
                          label_col="one_hot_list", 
                          **kwargs):
    # flow_from_dataframe returns a dataframeiterator
    dfi = super(ImageDataGenerator, self).flow_from_dataframe(
      dataframe, 
      ## x_col and y_col are both present on the dataframe, but bullshit to trick imagedatagenerator into letting us get the dataframeiterator
      x_col="image_path", y_col="one_hot_list", 
      **kwargs)
    dfi.dataframe = dataframe
    dfi.image_col = image_col
    dfi.text_cols = text_cols
    dfi.label_col = label_col
    # override this method just for this instance
    dfi._get_batches_of_transformed_samples = types.MethodType(batch_generation_override, dfi)
    # add text combination method
    if combination_type is CombinationType.SUM:
      dfi.vectorize = self.sum_text
    else:
      dfi.vectorize = self.concat_text
    return dfi