from src.data_generators import *

class InferenceDataGenerator(ImageTextDataGenerator):
  def __init__(self, vectorizer, **kwargs):
    self.vectorizer = vectorizer
    super().__init__(vectorizer, **kwargs)
    
  def flow_from_dataframe(self, 
                          dataframe, 
                          combination_type=CombinationType.SUM, 
                          image_col="image_path", 
                          text_cols=["title_and_description"], 
                          label_col="one_hot_list", 
                          class_mode=None,
                          shuffle=False,
                          **kwargs):
    # flow_from_dataframe returns a dataframeiterator
    dfi = super(ImageDataGenerator, self).flow_from_dataframe(
      dataframe, 
      ## x_col and y_col are both present on the dataframe, but bullshit to trick imagedatagenerator into letting us get the dataframeiterator
      x_col="image_path", 
      y_col="one_hot_list", 
      class_mode=None,
      shuffle=False,
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
