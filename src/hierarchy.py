import numpy as np
from tensorflow.python.keras import backend as K
import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

class Hierarchy:
  """
  Loss/Metric implementation of "A hierarchical loss and its problems when classifying 
  non-hierarchically" (https://arxiv.org/abs/1709.01062)
  
  The paper describes a tree-based algorithm for computing the "win" associated with a 
  prediction that takes the hierarchical structure of the data into consideration.
  
  To compute this with matrices, we first build an NxN adjacency matrix, A, where 
  A[i,j] = 1 if the jth node of the taxonomy is a descendent of the ith node, and 0
  otherwise. 
  
  Next, we construct N N-dimensional mask coefficient vectors. The ith element of the
  jth vector is 2^-m where m is 1 + the distance from the root of the tree, 2^-(m-1) if
  i=j and 0 otherwise.
  
  Letting q = A . p, where p is the predicted label and is a discrete probability
  distribution over the nodes, the ith element of q will have the correct value for
  the ith node as described in the propagating probabilities algorithm in the paper.
  
  The dot product of q and w where w is the win mask associated with the correct label
  is the quantity as described in the computing winnings algorithm in the paper.
  
      
  Example Usage:
    # first, load the taxonomy:
    taxonomy = load_taxonomy()
    
    # create the hierarchy—this will rebuild the taxonomy to put leaf nodes first
    # if that's not already the case:
    H = Hierarchy(taxonomy)
    
    # load your dataset:
    X, y = load_dataset(taxonomy)
    
    # prune off non-leaf nodes from the 1-hot encoded label:
    yH = y[:,:H.num_leaves]
    
    # create the metric instance and the loss function
    metric = HierarchyMetric(H)
    loss = hierarchical_loss(H)

    # compile your model
    model.compile(optimizer=<your optimizer>, loss=loss, metrics=[metric, CustomMetric()])
    
    # pass the pruned labels to fit
    model.fit(X, yH)
  """
  def __init__(self, taxonomy, id_key="id", parent_key="parent", children_key="children", selection_func=None):
    """Constructor
    
    Args:
      taxonomy: an N-length list of nodes with id_key: <index in taxonomy> and 
                parent_key: <index of parent> | None
      id_key: key for the index in the taxonomy
      parent_key: key for the parent index
      children_key: key for the list of child node indexes
      selection_func: (Optional) a function for selecting the correct index given a label
    """
    self.taxonomy = taxonomy
    self.id_key = id_key
    self.parent_key = parent_key
    self.children_key = children_key
    self.selection_func = selection_func
    self.leaf_key = "is_leaf"
    self._rebuild_taxonomy()
    self._build_adjacency_matrix()
    self._build_win_masks()
    self._build_selection_masks()
    self._build_padding_vector()
  
  @property
  def N(self):
    """Total number of nodes in the taxonomy"""
    return len(self.taxonomy)
  
  @property
  def num_leaves(self):
    """The number of leaf nodes in the taxonomy—this is the encoded label size"""
    return len([node for node in self.taxonomy if node[self.leaf_key]])
  
  @property
  def size(self):
    return (self.N, self.N)
  
  def new_id(self, old_id):
    return self.old_to_new_map[old_id]
  
  def encode(self, idx, is_old_id=False):
    """One hot encodes the node at idx as a numpy array and a tensorflow tensor"""
    if is_old_id:
      idx = self.new_id(idx)
    node = self.taxonomy[idx]
    assert node[self.leaf_key], "node is not a leaf in the taxonomy"
    encoding = np.zeros(self.num_leaves)
    encoding[idx] = 1
    return (encoding, K.variable([encoding]))
  
  def compute_win(self, y_true, y_pred, to_numpy=False):
    if self.N > self.num_leaves:
      # if there are more leaf nodes than total nodes in the hierarchy (should always be the case,
      # but allowed to work either way) then pad with a zero for each non-leaf node in the taxonomy
      y_true = self._pad(y_true)
      y_pred = self._pad(y_pred)
    # propagate the probabilities (algo 1)
    propagated_probabilities = K.dot(self.A, K.transpose(y_pred))
    # find the index from the actual label
    win_idx = self.select_correct_idx(y_true)
    # find the mask associated with that label
    win_mask = tf.gather(self.W, win_idx)
    # win is q . w (algo 2)
    win = K.batch_dot(win_mask, K.transpose(propagated_probabilities))
    # win is in [0.5,1], remap to [0,1]:
    remapped = 2 * (win - 0.5)
    if to_numpy:
      remapped = K.reshape(remapped, []).numpy()
    return remapped
  
  def select_correct_idx(self, y_true):
    if self.selection_func is not None:
      return self.selection_func(y_true)
    # default to expecting 1-hot and just choosing the max value
    return K.argmax(y_true)
  
  def select_best_leaf(self, y_pred):
    if self.N > self.num_leaves:
      # if there are more leaf nodes than total nodes in the hierarchy (should always be the case,
      # but allowed to work either way) then pad with a zero for each non-leaf node in the taxonomy
      y_pred = self._pad(y_pred)
    # propagate the probabilities (algo 1)
    propagated_probabilities = K.transpose(K.dot(self.A, K.transpose(y_pred)))
    # grab the mask vector for root and repeat it <batch size> times
    root = K.repeat(self.root, K.shape(y_pred)[0])
    # reshape into (<batch size>, N)
    predictions = K.reshape(root, (K.shape(y_pred)[0],))
    # each branch will walk futher out toward leaf nodes (and loops on leaf nodes)
    for _ in range(self.depth):
      predictions = self._branch(propagated_probabilities, predictions)
    return predictions
  
  @property
  def depth(self):
    # longest lineage (root to leaf)
    return max([len(list(self._lineage(x[self.id_key]))) for x in self.taxonomy])
  
  def _pad(self, y):
    if self.N > self.num_leaves:
      # pads the encoding with zeros in the place of non-leaf nodes
      # cast in case our labels are ints
      y = tf.cast(y, self.p.dtype)
      P = K.tile(self.p, (K.shape(y)[0],1))
      return K.concatenate((y, P))
  
  def _branch(self, probabilities, indices):
    # choose the child categories with the max probability value
    S = tf.gather(self.S, indices)
    branches = S * probabilities
    return K.argmax(branches, axis=-1)
  
  def _lineage(self, idx):
    node = self.taxonomy[idx]
    while self.parent_key in node and node[self.parent_key] is not None:
      # iterate until we reach the root
      yield node[self.parent_key]
      node = self.taxonomy[node[self.parent_key]]
  
  def _build_adjacency_matrix(self):
    A = np.zeros(self.size)
    for node in self.taxonomy:
      idx = node[self.id_key]
      A[idx, idx] = 1 # itself
      for ancestor in self._lineage(idx):
        # set a 1 so that all ancestors accumulate from idx's prediction value
        A[ancestor, idx] = 1
    self.A = K.constant(A)
  
  def _build_padding_vector(self):
    self.p = K.zeros((1, self.N - self.num_leaves))
  
  def _build_win_masks(self):
    W = [None] * self.N
    for node in self.taxonomy:
      idx = node[self.id_key]
      lineage = [idx] + [ancestor for ancestor in self._lineage(idx)]
      n = len(lineage)
      w = np.zeros(self.N)
      # this gets set out here because we double up on the leaf
      w[idx] = 2**-n
      for ancestor in lineage:
        # again, += to add to the value for the current leaf (we prepended the leaf node to the lineage)
        w[ancestor] += 2**-n
        n -= 1
      W[idx] = w
    self.W = K.constant(W)
  
  def _build_selection_masks(self):
    S = [None] * self.N
    for node in self.taxonomy:
      idx = node[self.id_key]
      s = np.zeros(self.N)
      if not node[self.leaf_key]:
        for child_idx in node[self.children_key]:
          s[child_idx] = 1
      else:
        s[idx] = 1
      S[idx] = s
    self.S = K.constant(S)
    
    root_idx = self.taxonomy[list(self._lineage(0))[-1]][self.id_key]
    self.root = K.constant([[root_idx]], dtype='int32')
  
  # restructures the order in the taxonomy list so that all leaf nodes are first
  def _rebuild_taxonomy(self):
    self.old_to_new_map = {}
    # flag nodes as leaves:
    for node in self.taxonomy:
      children = node[self.children_key]
      if children is None or len(children) == 0:
        node[self.leaf_key] = True
      else:
        node[self.leaf_key] = False
        # in case we don't have parent defined (which isn't strictly necessary 
        # like with children), add it:
        for child_idx in children:
          child = self.taxonomy[child_idx]
          child[self.parent_key] = node[self.id_key]
    # split on this:
    leaves   = [node for node in self.taxonomy if node[self.leaf_key]]
    branches = [node for node in self.taxonomy if not node[self.leaf_key]]
    
    # concat
    tax = leaves + branches
    
    # assign its new id
    for idx, node in enumerate(tax):
      new_key = "new"
      node[new_key] = idx
      self.old_to_new_map[node[self.id_key]] = idx
    
    # remap all relationships
    for node in tax:
      if self.parent_key in node:
        # grab the node by its old index, and assign the 
        # parent relationship with the new index on the child
        node[self.parent_key] = self.taxonomy[node[self.parent_key]][new_key]
      if not node[self.leaf_key]:
        # do the same thing for any children references
        node[self.children_key] = [self.taxonomy[child][new_key] for child in node[self.children_key]]
    
    # assign the new id, and delete the now useless new key
    for node in tax:
      node[self.id_key] = node[new_key]
      del node[new_key]
    
    # we want to mutate the passed in taxonomy, so manually remove and add:
    while len(self.taxonomy) > 0:
      self.taxonomy.pop()
    for node in tax:
      self.taxonomy.append(node)