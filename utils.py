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



from sklearn.cluster import KMeans
def cluster_and_evaluate(X, random_state, y_true, num=2):
    
    kmeans = KMeans(n_clusters=num, random_state=random_state,n_init=10).fit(X)
    y_pred = kmeans.predict(X)
    correct = [1 if item1 == item2 else 0 for (item1,item2) in zip(y_true, y_pred) ]
    alignment = sum(correct)/float(len(correct))
    max_alignment = max(alignment, 1 - alignment)
   
    print(f'Alignment: {max_alignment}')
    
    return kmeans, y_pred, X, max_alignment


#Removes the projection of vector 1 on vector 2
def remove_vector_projection(vector1, vector2):
    projection= (np.dot(vector1,vector2) / np.linalg.norm(vector2))*vector2
    difference = vector1 - projection
    return difference

# Function to compute the gender bias of a word. 
# Outputs a dictionary with words as keys and gender bias as values
def compute_bias(dict_vectors, he_embedding, she_embedding):
    gender_bias = {}
    for word in dict_vectors.keys():
        vector = dict_vectors[word]
        gender_bias[word] = cosine_similarity(vector, she_embedding) - cosine_similarity(vector, he_embedding)
    return gender_bias


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

def get_bias_score_df_from_list(bias_scores_original, debiased_scores, word_list, vocab_cleaned, debiased_vocab_limited): 
    word_set = set(vocab_cleaned)
    word_set_debiased = set(debiased_vocab_limited)
    scores = {}
    for word in word_list: 
        if word in word_set: 
            scores[word] = {"original_score": bias_scores_original[word]}
        if word in word_set_debiased: 
            scores[word] = scores.get(word, {})
            scores[word]["debiased_score"] = debiased_scores[word]
  
    full_scores_df = pd.DataFrame.from_dict(scores, orient='index').reset_index()

    full_scores_df_long = pd.melt(full_scores_df, id_vars=["index"], value_vars=["original_score", "debiased_score"])
    full_scores_df_long["value"].astype("float")

    return full_scores_df_long
