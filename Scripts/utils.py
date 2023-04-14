import numpy as np
import pandas as pd
from operator import itemgetter

# Cosine Similarity measures the similarity of two word vectors. 
#        Ie. vectors are similar when the angle between them is close to 0 (cosine close to 1).
def cosine_similarity(v, w):
    
    dot_product=np.dot(v,w)
    product_norms=np.linalg.norm(v)*np.linalg.norm(w)
    cosine_similarity = dot_product/product_norms
    
    return cosine_similarity



from sklearn.decomposition import PCA      # PCA library
import pandas as pd    
# get main PCA components on descentralized word embeddings
def get_main_pca_all(word_vectors):
    #Generating the descentralized word embeddings- 
    word_vectors_mean = np.mean(np.array(word_vectors), axis=0)
    word_vectors_hat = np.zeros(word_vectors.shape).astype(float)

    for i in range(len(word_vectors)):
        word_vectors_hat[i, :] = word_vectors[i, :] - word_vectors_mean

    main_pca = PCA()
    main_pca.fit(word_vectors_hat)
    
    return main_pca


def WEAT():
    pass



def finding_k_nearest_neighbors():
    pass


def extract_vectors(words, vectors, w2i):
    
    X = [vectors[w2i[x],:] for x in words]
    
    return X

#Removes the projection of vector 1 on vector 2
def remove_vector_projection(vector1, vector2):
    projection= (np.dot(vector1,vector2) / np.linalg.norm(vector2))*vector2
    difference = vector1 - projection
    return difference

def normalize(wv):
    # normalize vectors
    norms = np.apply_along_axis(np.linalg.norm, 1, wv)
    wv = wv / norms[:, np.newaxis]
    return wv

def prepare_def_sets_subspace(list_def_sets):
  def_sets={i: v for i, v in enumerate(list_def_sets)}
  return def_sets

def get_words_from_pairs(definitional_pairs):
  # Turning the pairs into words to add afterwards to the vocabulary 
  definitional_pairs=[]
  for pair in definitional_pairs: 
    for word in pair: 
      definitional_pairs.append(word)
  return definitional_pairs

def getting_biased_words(gender_bias_original, definitional_pairs, size, word2idx):
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
  gendered_words=get_words_from_pairs(definitional_pairs)
  c_vocab = list(set(male_words + female_words + [word for word in gendered_words if word in word2idx]))
  c_w2i = dict()
  for idx, w in enumerate(c_vocab):
    c_w2i[w] = idx
  return c_w2i, c_vocab, female_words, male_words, y_true

from sklearn.decomposition import PCA
def perform_PCA_pairs(pairs, word_vectors, word2index, num_components=2):
        
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