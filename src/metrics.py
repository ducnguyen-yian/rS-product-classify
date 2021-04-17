import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K

def custom_metric(y_true, y_pred):
  idx_true = K.argmax(y_true, -1)
  idx_pred = K.argmax(y_pred, -1)
  return K.mean(math_ops.equal(idx_true, idx_pred), axis=-1)
  
class CustomMetric(MeanMetricWrapper):
  def __init__(self, name='custom', dtype=None):
    super(CustomMetric, self).__init__(custom_metric, name, dtype=dtype)

class HierarchicalAccuracy(MeanMetricWrapper):
  def __init__(self, hierarchy, name='hierarchical_accuracy', dtype=None):
    def metric(y_true, y_pred):
      idx_pred = hierarchy.select_best_leaf(y_pred)
      idx_true = K.argmax(y_true, -1)
      return K.mean(math_ops.equal(idx_true, idx_pred), axis=-1)
    super(HierarchicalAccuracy, self).__init__(metric, name, dtype=dtype)
    
class HierarchyMetric(MeanMetricWrapper):
  def __init__(self, hierarchy, name='hierarchy', dtype=None):
    def hierarchy_metric(y_true, y_pred):
      return K.mean(hierarchy.compute_win(y_true, y_pred), axis=-1)
    super(HierarchyMetric, self).__init__(hierarchy_metric, name, dtype=dtype)

class KHotTopMetric(MeanMetricWrapper):
  def __init__(self, taxonomy, id_key="id", parent_key="parent", name='khot_top', dtype=None):
    self.taxonomy = taxonomy
    self.id_key = id_key
    self.parent_key = parent_key
    self._build_label_matrix()
    super(KHotTopMetric, self).__init__(self.metric, name, dtype=dtype)
  
  def metric(self, y_true, y_pred):
    yL_true = K.dot(self.L, K.transpose(y_true))
    yL_pred = K.dot(self.L, K.transpose(y_pred))
    idx_true = K.argmax(yL_true, 0)
    idx_pred = K.argmax(yL_pred, 0)
    return K.mean(math_ops.equal(idx_true, idx_pred), axis=-1)
  
  def _build_label_matrix(self):
    N = len(self.taxonomy)
    L = np.zeros((N, N))
    for node in self.taxonomy:
      row = node[self.id_key]
      while self.parent_key in node and node[self.parent_key] is not None:
        col = node[self.id_key]
        L[row,col] = 1
        node = self.taxonomy[node[self.parent_key]]
    self.X = L
    self.L = K.constant(L)
    
class FallbackAccuracy(MeanMetricWrapper):
  def __init__(self, hierarchy, threshold=0.5, name='fallback', dtype=None):
    self.hierarchy = hierarchy
    self.threshold = threshold
    super(FallbackAccuracy, self).__init__(self.fallback_metric, name, dtype=dtype)
  
  def fallback_metric(self, y_true, y_pred):
    #grab the most confident prediction
    predictions=K.max(y_pred, axis=-1)
    
    #fill a tensor with our threshold_value
    threshold_tensor=tf.fill(tf.shape(predictions), self.threshold)
    
    #Are we confident in our prediction?
    threshold_high=predictions>threshold_tensor
    threshold_high=tf.cast(threshold_high, tf.int32)
    
    #Do we have low confidence in our prediction?
    threshold_low=predictions<=threshold_tensor
    threshold_low=tf.cast(threshold_low, tf.int32)
    
    idx_true = K.argmax(y_true, -1)
    idx_pred = K.argmax(y_pred, -1)
    
    #For our confident predictions, compare the top prediction to the label of the true value
    high_correct=math_ops.equal(idx_true, idx_pred)
    high_correct=tf.cast(high_correct, tf.int32)
    
    #For our less confident predictions, grab the top 2 most confident predictions
    _, max_pred = tf.math.top_k(y_pred, k=2)

    #Gather the lineages of those top 2 predictions using the transpose of the hierarchy's adjaency matrix because the adjacency only points from ancestor to descendant
    lineages = tf.gather(K.transpose(self.hierarchy.A), max_pred)
    lineages = K.cast(lineages, tf.int32)
    
    #Grab the first two columns of this matrix
    fallback = tf.bitwise.bitwise_and(lineages[:,0], lineages[:,1])
    
    #Gather the lineage of the true value
    actual = tf.gather(K.transpose(self.hierarchy.A), K.argmax(y_true))
    actual = K.cast(actual, tf.int32)
    
    #Multiply the two together
    overlap_score = K.batch_dot(fallback, actual)

    #Are either of the top 2 predictions in the lineage of the true value? If so, overlap_score should be >1 and we count the result as correct
    low_correct = overlap_score > 1
    low_correct=tf.cast(low_correct, tf.int32)
    low_correct=tf.squeeze(low_correct)
    
    #results for the high confidence predictions
    high_accuracy=tf.math.multiply(threshold_high, high_correct)
    
    #results for the low confidence predictions
    low_accuracy=tf.math.multiply(threshold_low, low_correct)
    
    # total accuracy vector
    correct=high_accuracy+low_accuracy
    
    #return batch accuracy value
    return K.mean(K.cast(correct, tf.float32))