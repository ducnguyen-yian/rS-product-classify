from tensorflow.python.keras import backend as K

def hierarchical_loss(hierarchy):
  def loss(y_true, y_pred):
    win = hierarchy.compute_win(y_true, y_pred)
    return -K.log(win)
  return loss