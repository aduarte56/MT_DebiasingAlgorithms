import numpy as np
import scipy
from tqdm import tqdm
import string
import warnings
import random
from typing import Dict, List, Tuple, Union



class Classifier():

    def __init__(self):

        pass
   
    def train(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set
        """
        raise NotImplementedError

   
    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """
        raise NotImplementedError



class SKlearnClassifier(Classifier):

    def __init__(self, m):
        """
        :param m: sklearn model
        """
        self.model = m

    def train_network(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set / Person's R in the case of regression
        """
        self.model.fit(X_train, Y_train)
        score = self.model.score(X_dev, Y_dev)
        return score

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """
        w = self.model.coef_
        if len(w.shape) == 1:
                w = np.expand_dims(w, 0)

        return w

def get_rowspace_projection(W):
    """
    :param W: np array of shape (input_dim, output_dim)
    :return 
        :return: final weights of the model, as np array
    """
    if np.allclose(W, 0): #Returns True if two arrays are element-wise equal within a tolerance.
      w_basis = np.zeros_like(W.T) #Return an array of zeros with the same shape and type as the transpose of the given array.
    else:
      w_basis = scipy.linalg.orth(W.T) # orthogonal basis for the columns of W.T (rows of W)
    P_W = w_basis.dot(w_basis.T) # orthogonal projection matrix on W's rowspace because vectors are orthonormal.
    return P_W 
    

def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):
    """
    :param: rowspace_projection_matrices: list of projection matrices on the rowspaces of the classifiers
    :param: input_dim: dimension of the input space 
    :return 
        :return: projection matrix to the intersection of the nullspaces of the classifiers
    """
    I = np.eye(input_dim) #Return a 2-D array with ones on the diagonal and zeros elsewhere.
    Q = np.sum(rowspace_projection_matrices, axis = 0)
    P = I -get_rowspace_projection(Q)

    return P

def get_debiasing_projection(classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
                             min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
                             Y_dev: np.ndarray, is_autoregressive: bool,
                             Y_train_main = None, Y_dev_main=None, dropout_rate=0) -> np.ndarray:
    """
    :param classifier_class: class of the classifier to use
    :param cls_params: parameters of the classifier
    :param num_classifiers: number of classifiers to train
    :param input_dim: dimension of the input space
    :param min_accuracy: minimum accuracy on the dev set for a classifier to be considered
    :param X_train: training set
    :param Y_train: training labels
    :param X_dev: dev set
    :param Y_dev: dev labels
    :param is_autoregressive: whether to use autoregressive debiasing
    :param Y_train_main: training labels of the main task
    :param Y_dev_main: dev labels of the main task
    :param dropout_rate: dropout rate
    :return: projection matrix to the intersection of the nullspaces of the classifiers
    """

    I = np.eye(input_dim)

    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    rowspace_projections = []
    Ws = []

    pbar = tqdm(range(num_classifiers)) #gets the progress bar for num_classifiers
    for i in pbar:

      clf = SKlearnClassifier(classifier_class(**cls_params))
      dropout_scale = 1./(1 - dropout_rate + 1e-6)
      dropout_mask = (np.random.rand(*X_train.shape) < (1-dropout_rate)).astype(float) * dropout_scale

      relevant_idx_train = np.ones(X_train_cp.shape[0], dtype=bool)
      relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

      acc = clf.train_network((X_train_cp * dropout_mask)[relevant_idx_train], Y_train[relevant_idx_train], X_dev_cp[relevant_idx_dev], Y_dev[relevant_idx_dev])
      pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
      if acc < min_accuracy: continue

      W = clf.get_weights()
      Ws.append(W)
        
      P_rowspace_wi = get_rowspace_projection(W) # projection to W's rowspace
      rowspace_projections.append(P_rowspace_wi)

      if is_autoregressive:

          """
            to ensure numerical stability, explicitly project to the intersection of the nullspaces found so far (instaed of doing X = P_iX,
            which is problematic when w_i is not exactly orthogonal to w_i-1,...,w1, due to e.g inexact argmin calculation).
            """
          # use the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
          # N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))

          P = get_projection_to_intersection_of_nullspaces(
              rowspace_projections, input_dim)
          # project

          X_train_cp = (P.dot(X_train.T)).T
          X_dev_cp = (P.dot(X_dev.T)).T

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P, rowspace_projections, Ws




      #Functions for gender debiasing
    
def getting_classes_for_INLP(gender_vector, model, n = 2500):
    """"
    :param gender_vector : gender direction vector
    :param model : embeddings model
    :param n : number of words to be returned
    :return: groups of biased words: group1, group2, neut
    """
    group1 = model.similar_by_vector(gender_vector, topn = n, restrict_vocab=None)
    group2 = model.similar_by_vector(-gender_vector, topn = n, restrict_vocab=None)
    
    all_sims = model.similar_by_vector(gender_vector, topn = len(model.vectors), restrict_vocab=None)
    eps = 0.03
    idx = [i for i in range(len(all_sims)) if abs(all_sims[i][1]) < eps]
    samp = set(np.random.choice(idx, size = n))
    neut = [s for i,s in enumerate(all_sims) if i in samp]
    return group1, group2, neut