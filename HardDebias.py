"""
@author: angeladuartepardo

This script contains the functions to perform the Hard-Debias algorithm following the approach of Bolukbasi et al. (2016),
Manzini et al. (2019), Cheng et al. (2019). Their work was addapted to the problem. Several boolean parameters are included 
to validate what happens to the embeddings while debiasing. 
The main functions call on the utils.py script to perform the PCA and remove the projection of a vector on a subspace.

This file can also be imported as a module and contains the following
functions:
    * find_bias_direction - finds the bias direction
    * identify_bias_subspace - identifies the bias subspace
    * neutralize_words - neutralizes words
    * equalize_words - equalizes words
    * hard_debias - performs the hard debias algorithm
    * hard_debias_intersectional - performs the intersectional hard debias algorithm
    * get_debiased_embeddings - returns the debiased embeddings dictionary word as key, vector as value. 
"""


import numpy as np
import utils
from ProcessingEmbeddings import get_debiased_dict
from sklearn.decomposition import PCA


def identify_bias_subspace(vector_dict, def_sets, subspace_dim, centralizing=True):
    """
    This identifying subspace function follows both Bolukbasi's Hard-Debias algorithm and Manzini's 
    intersectional hard debias algorithm.     
    ----------
    Parameters: 
    vector_dict - dictionary containing words as keys and their embeddings as values
    def_sets - sets of words that represent the bias subspace
    subspace_dim - number of components for PCA. It is also a parameter that provides the number of vectors defining the bias subspace
    centralizing - boolean that indicates whether the vectors should be centralized before running PCA
    """
    # calculate means of defining sets
    means = {}
    matrix = []
    #Loop through each set to get a list of all the words that are also in the vocabulary
    for k, v in def_sets.items():
      wSet = []
      for w in v:
        try:
          wSet.append(vector_dict[w])
        except KeyError as e:
          pass
      if wSet: 
        set_vectors = np.array(wSet)
        #Centralized vectors by subtracting their mean
        if centralizing:
          mean_vector = np.mean(set_vectors, axis=0)
          means[k]=mean_vector
          diffs = set_vectors - mean_vector
          matrix.append(diffs)
        else:
          if len(wSet)==2:
            diffs=np.array([wSet[0]-wSet[1]])
            matrix.append(diffs)

    matrix = np.concatenate(matrix)
    print('Length of vectors set:',len(matrix))  

    matrix = np.array(matrix)
    #Run PCA on the vectors of the definitional sets. 
    pca = PCA(n_components=subspace_dim)
    print('Running PCA with', subspace_dim, 'components')
    pca.fit(matrix)

    return pca.components_

def neutralize_words(vocab_partial, vectors, w2i_partial, bias_direction): 
    """
    This function neutralizes words by removing their projection on their bias subspace. 
    In this way, the resulting vectors are orthogonal to the bias subspace.
    As a result, the function outputs the debiased vectors.
    The approach follows Bolukbasi's Hard-Debias algorithm 
    ----------
    Parameters: 
    vocab_partial - limited vocabulary
    vectors - word embeddings
    w2i_partial - dictionary that maps words to their indices in the vocabulary
    bias_direction - bias subspace
    """
    #Removes the bias component of words that should be neutral
    debiased_vectors = np.zeros((len(vocab_partial), len(vectors[0, :]))).astype(float)
    for i, words in enumerate(vocab_partial):
      u = vectors[w2i_partial[words], :]
      u = utils.remove_vector_projection(u, bias_direction)
      debiased_vectors[w2i_partial[words], :] = u
    return debiased_vectors




def equalize_words(vectors, vocab_partial, w2i_partial, equalizing_list, bias_direction):
    """
    This function equalizes with respect to the neutrality axis, this means that they are centered 
    with respect to the origin so that they are equidistant from the neutrality axis. 
    ----------
    Parameters: 
    vocab_partial - limited vocabulary
    vectors - word embeddings
    w2i_partial - dictionary that maps words to their indices in the vocabulary
    bias_direction - bias subspace
    equalizing_list - list of pairs of words that should be equalized
    """
    debiased_vectors = vectors
    candidates = set()
    for word1, word2 in equalizing_list:
      candidates.add((word1.lower(), word2.lower()))
      candidates.add((word1.title(), word2.title()))
      candidates.add((word1.upper(), word2.upper()))  
 
    for (a, b) in candidates:
      if (a in vocab_partial and b in vocab_partial):
        mean_c= (vectors[w2i_partial[a], :]-vectors[w2i_partial[b], :])/2
        mean_orth = utils.remove_vector_projection(mean_c, bias_direction)
        z = np.sqrt(abs(1 - np.linalg.norm(mean_orth))**2)
        if (vectors[w2i_partial[a], :] - vectors[w2i_partial[b],:]).dot(bias_direction) < 0:
          z = -z
        debiased_vectors[w2i_partial[a], :] = z * bias_direction + mean_orth
        debiased_vectors[w2i_partial[b], :] = -z * bias_direction + mean_orth
    return debiased_vectors

def hard_debias(wv, vector_dict_partial, w2i_partial, vocab_partial,
                 equalizing_list, def_sets, subspace_dim, normalize_dir=False, normalize=None, centralizing=True):
    """
    Hard debiasing algorithm following Bolukbasi's Hard-Debias algorithm. Calls other functions 
    to identify the bias subspace, neutralize words and equalize words.
    returns the debiased vectors. 
    ----------
    Parameters: 
    vocab_partial - limited vocabulary
    vectors - word embeddings
    w2i_partial - dictionary that maps words to their indices in the vocabulary
    bias_direction - bias subspace
    """    
    vectors=wv.copy()
    #Gender direction
    bias_direction=identify_bias_subspace(vector_dict_partial, def_sets, subspace_dim, centralizing=centralizing)
   
    if normalize_dir:
      bias_direction=utils.normalize(bias_direction)
    #Following Manzini
    if bias_direction.ndim == 2:
        bias_direction = np.squeeze(bias_direction)
    elif bias_direction.ndim != 2:
        raise ValueError("bias subspace should be either a matrix or vector")
    
   
    
    if str(normalize).lower()=="before":
      vectors= utils.normalize(vectors) #Following Andrew Ng's approach
      

    wv_debiased=neutralize_words(vocab_partial, vectors, w2i_partial, bias_direction)
    if str(normalize).lower()=="after":
      wv_debiased=utils.normalize(wv_debiased) #Following Bolukbasi
     
    wv_debiased=equalize_words(wv_debiased, vocab_partial, w2i_partial, equalizing_list, bias_direction)

    if str(normalize).lower()=="after":
      wv_debiased=utils.normalize(wv_debiased)#Following Bolukbasi
    
    debiased_dict=get_debiased_dict(wv_debiased, w2i_partial)
    return wv_debiased, vocab_partial, w2i_partial, debiased_dict


