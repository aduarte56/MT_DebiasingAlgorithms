from sklearn.decomposition import PCA      # PCA library
from itertools import product
import numpy as np
import pandas as pd
from operator import itemgetter
import itertools
from itertools import product


################################################
# OPERATIONS ON THE DEFINITIONAL SETS
#################################################

#Function to get the pairs of words from the equalizing sets
def get_pairs_from_equalizing_sets(def_sets):
    """"
    Gets the pairs of words from the equalizing sets
    ----
    :param def_sets: list of lists of words
    :return: list of pairs of words
    """
    data = {i: v for i, v in enumerate(def_sets)}
    #print(data)
    pairs = []
    for _, values in data.items():
	    #Get all possible combinations of pairs
	    for v1 in values:
		    for v2 in values:
                        s = set([v1, v2])
                        if(len(s) > 1 and not (v1 in pairs and v2 in pairs)):
                            pairs.append([v1, v2])
                      #     if (v1 in pairs or v2 in pairs):
                       #          print(v1,v2)
	#Remove duplicates
    pairs.sort()
    cleaned_pairs = list(k for k, _ in itertools.groupby(pairs))
    return cleaned_pairs


#Function to get the pairs of words from to different definitional sets.
def get_pairs(p1, p2):
    """"
    Gets the pairs of words from lists of lists of words
    ----
    :param p1: list of words
    :param p2: list of words
    :return: list of pairs of words
    """
    pairs = set()
    for v1, v2 in product(p1, p2):
        for val1, val2 in product(v1, v2):
            pairs.add((val1, val2))
    return list(pairs)


def prepare_def_sets_subspace(list_def_sets):
  """"
  Prepares the definitional sets for the subspace method
  ----
  :param list_def_sets: list of lists of words
  :return: dictionary of lists of words
  """
  def_sets = {i: v for i, v in enumerate(list_def_sets)}
  return def_sets


def get_words_from_pairs(definitional_pairs):
  """"  
  Gets the words from the pairs of words
  ----
  :param definitional_pairs: list of pairs of words
  :return: list of words
  """
  # Turning the pairs into words to add afterwards to the vocabulary
  definitional_list = []
  for pair in definitional_pairs:
    for word in pair:
      definitional_list.append(word)
  return definitional_list

################################################
# OPERATIONS ON WORD VECTORS
#################################################

# Cosine Similarity measures the similarity of two word vectors.
#        Ie. vectors are similar when the angle between them is close to 0 (cosine close to 1).
def cosine_similarity(v, w):
    """"
    Computes the cosine similarity between two vectors
    ----
    :param v: vector
    :param w: vector
    :return: cosine similarity
    """
    dot_product = np.dot(v, w)
    product_norms = np.linalg.norm(v)*np.linalg.norm(w)
    # 1e-6 is a small number to avoid division by zero
    cosine_similarity = dot_product/max(product_norms, 1e-6)

    return cosine_similarity

def extract_vectors(words, vectors, w2i):
    """"
    Extracts the vectors of the words
    ----
    :param words: list of words
    :param vectors: word vectors
    :param w2i: dictionary of words and their indices
    :return: list of vectors
    """
    
    X = [vectors[w2i[x],:] for x in words]
    
    return X

#Removes the projection of vector 1 on vector 2
def remove_vector_projection(vector1, vector2):
    """"
    Removes the projection of vector 1 on vector 2
    ----
    :param vector1: vector
    :param vector2: vector
    :return: difference between the two vectors
    """
    projection= (np.dot(vector1,vector2) / np.linalg.norm(vector2))*vector2
    difference = vector1 - projection
    return difference

def normalize(wv):
    """"
    Normalizes the vectors
    ----
    :param wv: word vectors
    :return: normalized word vectors
    """
    # normalize vectors
    norms = np.apply_along_axis(np.linalg.norm, 1, wv)
    wv = wv / norms[:, np.newaxis]
    return wv

from sklearn.decomposition import PCA
def perform_PCA_pairs(pairs, word_vectors, word2index, num_components=2):
    """"
    Performs PCA on the pairs of words
    ----
    :param pairs: list of pairs of words
    :param word_vectors: word vectors
    :param word2index: dictionary of words and their indices
    :param num_components: number of components
    :return: PCA of the pairs of words
    """
    vector_matrix = []
    count = 0
    
    if type(pairs[0]) is list:
        #Centering the gendered pairs (so that the )
        for feminine_word, masculine_word in pairs:
            if not (feminine_word in word2index and masculine_word in word2index): 
              continue
            center = (word_vectors[word2index[feminine_word], :] + word_vectors[word2index[masculine_word], :])/2
            vector_matrix.append(word_vectors[word2index[feminine_word], :] - center)
            vector_matrix.append(word_vectors[word2index[masculine_word], :] - center)
            count += 1
            
    else:
        for word in pairs:
            if not (word in word2index):
               continue
            vector_matrix.append(word_vectors[word2index[word], :])
            count += 1
        
        vector_matrix = np.array(vector_matrix)
        vectors_mean = np.mean(np.array(vector_matrix), axis=0)
        wv_hat = np.zeros(vector_matrix.shape).astype(float)
    
        for i in range(len(vector_matrix)):
            wv_hat[i, :] = vector_matrix[i, :] - vectors_mean
        vector_matrix = wv_hat
            
    matrix = np.array(vector_matrix)
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    #print('pairs used in PCA: ', count)
    #print(pca.explained_variance_ratio_)
    return pca


# get main PCA components on descentralized word embeddings
def get_main_pca_all(word_vectors):
    """"
    Gets the main PCA components on descentralized word embeddings
    ----
    :param word_vectors: word vectors
    :return: main PCA components on descentralized word embeddings
    """
    #Generating the descentralized word embeddings-
    word_vectors_mean = np.mean(np.array(word_vectors), axis=0)
    word_vectors_hat = np.zeros(word_vectors.shape).astype(float)

    for i in range(len(word_vectors)):
        word_vectors_hat[i, :] = word_vectors[i, :] - word_vectors_mean

    main_pca = PCA()
    main_pca.fit(word_vectors_hat)

    return main_pca


############################################
# PREPARATION FOR EVALUATION,
############################################

def getting_biased_words(gender_bias_original, definitional_pairs, size, word2idx):
    """  
    Gets the biased words
    ----
    :params gender_bias_original: dictionary of biased words and their bias
    :params definitional_pairs: list of pairs of words
    :params size: size of the biased words
    :params word2idx: dictionary of words and their indices
    :return: list of biased words
    """
    # Sorting gender_bias_original in the ascending order so that all the female biased words will be at the start and
    # all the male biased words will be at the end.
    biased_words_sorted = sorted(gender_bias_original.items(), key=itemgetter(1))

    # Considering 1000 male and 1000 female biased words.
    # `size` can be anything, the authors mentioned in the paper that they took 500 male and 500 female top biased words.
    # But we were not able to get the same results by taking 500 male and 500 female top biased words so we considered
    # 1000 male and 1000 female top biased words based on thier code.
    male_words = [word for word, bias in biased_words_sorted[:size]]
    female_words = [word for word, bias in biased_words_sorted[-size:]]

    y_true = [0]*size + [1]*size
    gendered_words = get_words_from_pairs(definitional_pairs)
    c_vocab = list(set(female_words + male_words +
                 [word for word in gendered_words if word in word2idx]))
    c_w2i = dict()
    for idx, w in enumerate(c_vocab):
        c_w2i[w] = idx
    return c_w2i, c_vocab, female_words, male_words, y_true
