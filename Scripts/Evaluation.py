from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
from Scripts.utils import cosine_similarity
import pandas as pd

#############################
# ANALOGIES
#############################
#function to produce analogies between words: A is to B as X is to Y


def get_term_analogies(dict_vect, worda, wordb, wordx, include_triplet=False):
    """
    Following Bolukbasi et al. the metric to get the analogies is cos(worda-wordb,wordx -wordy). 
    The following method is an adaptation of the work of jmyao17. 
    """
    #initialize the maximum-similarity
    max_similarity = -10
    #get the vocabulary
    vocab = list(dict_vect.keys())

    #loop over each word in the vocabulary
    for word in vocab:
        # following Nissim et al. one problem with analogies is that they often don't allow for the forth word to be among the first three.
        # To avoid this, there is a boolean parameter to include the triplet of words in the analogy or not.
        if include_triplet == True:
            if word in [worda.lower(), wordb.lower(), wordx.lower()]:
                pass
            else:
                continue
        #Calculate the cosine si
        similarity = cosine_similarity(
            dict_vect[wordb.lower()]-dict_vect[worda.lower()],   dict_vect[word.lower()]-dict_vect[wordx.lower()])

        #if the similarity is higher than the maximum similarity, update the maximum similarity and the best word
        if similarity > max_similarity:
            max_similarity = similarity
            best_word = word

    return best_word, max_similarity



#############################
# PRE/POST BIAS SCORES
#############################

# Function to compute the gender bias of a word.
# Outputs a dictionary with words as keys and gender bias as values
def compute_gender_simple_bias(dict_vectors, he_embedding, she_embedding):
    gender_bias = {}
    for word in dict_vectors.keys():
        vector = dict_vectors[word]
        gender_bias[word] = cosine_similarity(
            vector, she_embedding) - cosine_similarity(vector, he_embedding)
    return gender_bias


def compute_direct_bias(dict_vectors, word_list, bias_subspace):
    directBias = {}
    word_list = set(word_list)
    for word in dict_vectors.keys():
        if word not in word_list:
            continue
        vector = dict_vectors[word]
        #directBias[word] = np.linalg.norm(
        #cosine_similarity(vector, np.squeeze(bias_subspace)))
        directBias[word] = cosine_similarity(vector, np.squeeze(bias_subspace))
    return directBias

#function to compute the average bias of the words in the neutral_words list
def compute_average_bias(dict_vectors, words_list, bias_subspace):
    directBias = compute_direct_bias(dict_vectors, words_list, bias_subspace)
    #get the norm of the bias for each word first
    for word in directBias.keys():
        directBias[word] = np.linalg.norm(directBias[word])
    #then compute the average
    average_bias = np.mean(list(directBias.values()))
    return average_bias


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

    full_scores_df = pd.DataFrame.from_dict(
        scores, orient='index').reset_index()

    full_scores_df_long = pd.melt(full_scores_df, id_vars=["index"], value_vars=[
                                  "original_score", "debiased_score"])
    full_scores_df_long["value"].astype("float")

    return full_scores_df_long


#############################
# CLUSTERING
#############################

def cluster_and_evaluate(X, random_state, y_true, num=2):
    
    kmeans = KMeans(n_clusters=num, random_state=random_state,n_init=10).fit(X)
    y_pred = kmeans.predict(X)
    correct = [1 if item1 == item2 else 0 for (item1,item2) in zip(y_true, y_pred) ]
    alignment = sum(correct)/float(len(correct))
    max_alignment = max(alignment, 1 - alignment)
   
    print(f'Alignment: {max_alignment}')
    
    return kmeans, y_pred, X, max_alignment

#############################
# RANDOM WORDS TEST
#############################
from collections import defaultdict

#the function uses gensim .most_similar() method to find the top-N most similar words to a given word
def finding_neighbors_before_after(word_list, original_model, debiased_model, topn=3):
    neighbors_per_word = defaultdict(dict)
    for word in word_list:
        words_before, _ = zip(*original_model.most_similar(word, topn=topn))
        words_after, _ = zip(*debiased_model.most_similar(word, topn=topn))
        neighbors_per_word[word]["before"] = words_before
        neighbors_per_word[word]["after"] = words_after
        print("----------------------------------")
        print("word: {}\n most-similar-before: {}\n most-similar-after: {}".format(word,
              words_before, words_after))
    return neighbors_per_word

#Get the embeddings of the neighbors from the words in the word_list
def get_embeddings_neighbors(keys,original_model, model_cleaned, topn):
    embedding_clusters = []
    db_embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        debiased_vectors=[]
        words = []
        for similar_word, _ in original_model.most_similar(word, topn=topn):
            words.append(similar_word)
            embeddings.append(original_model[similar_word])
            debiased_vectors.append(model_cleaned[similar_word])
           
        embedding_clusters.append(embeddings)
        #print('embeddings shape', np.array(embeddings).shape)
        db_embedding_clusters.append(debiased_vectors)
        word_clusters.append(words)
    embedding_clusters=np.array(embedding_clusters)
    db_embedding_clusters=np.array(db_embedding_clusters)

    return embedding_clusters, db_embedding_clusters, word_clusters


#Function to find the top-k most similar words to a given word using the cosine similarity


def get_topK_neighbors(word, dict_vect, vocab, vectors, w2i, k=10):
    k_neigh = {}
    list_neigh = []
    # extract the word vector for word w
    idx = w2i[word]
    chosen_vec = dict_vect[word]

    # compute cosine similarity between chosen_vec and all other words. Store in similarities list
    similarities = np.zeros(len(vocab))
    for i in range(len(vocab)):
        similarities[i] = cosine_similarity(chosen_vec, vectors[i])
    #similarities =[cosine_similarity(vectors.dot(chosen_vec)
    # sort similarities by descending order
    sorted_similarities = (similarities.argsort())[::-1]

    # choose topK
    best = sorted_similarities[:(k+1)]

    #create a list with the word and similarity score for each of the topK words
    k_neig_similarities = [(vocab[i], similarities[i])
                           for i in best if i != idx]
    k_neigh[word] = k_neig_similarities
    list_neigh = [vocab[i] for i in best if i != idx]
    return k_neigh, list_neigh

#get a dictionary with all the k-nearest neighbors for each word in the list
def get_k_nearest_neighbors(list_words, dict_vect, vocab, vectors, w2i, k=10):
    k_neigh ={}
    for w in tqdm(list_words):
        dict_neigh, _ = topK(w, dict_vect, vocab, vectors, w2i, k)
        k_neigh.update(dict_neigh)
    return k_neigh

#get a list of the neighbors for each word of the dictionary k_neigh


def get_list_neighbors(k_neigh):
    list_neigh = []
    for w in k_neigh.keys():
        list_neigh.append([i[0] for i in k_neigh[w]])
    return list_neigh

#function to get the frequency of the original neighbors among the 50 nearest neighbors of selected words


def get_frequency_original_neighbors(list_words, list_neigh, dict_vect_debiased, vocab_debiased, vectors_debiased, w2i_debiased, neighbours_num=50):

    scores = []
    for idx, word in tqdm(enumerate(list_words)):

        _, top = topK(word, dict_vect_debiased, vocab_debiased, vectors_debiased, w2i_debiased,
                      k=neighbours_num)

        count = 0

        for t in top:
            if t in list_neigh[idx]:
                print(t)
                count += 1

        scores.append([word, count, count/neighbours_num])
        #print(top)
    return scores


#############################
# MAC SCORES
#############################

from scipy import spatial
import numpy as np

#Function adapted from by Manzini et al. (2018)
def s_function_for_t_word(dict_vectors, target_word, attributes):
    attribute_vectors = np.array([dict_vectors[attribute]
	                               for attribute in attributes])
    target_vector = dict_vectors[target_word]
    cosine_distances = spatial.distance.cdist([target_vector], attribute_vectors, metric='cosine').flatten()
    return cosine_distances.mean()


def multiclass_evaluation_MAC(dict_vectors, targets_list, attributes):
	targets_eval = []
	for targetSet in targets_list:
		for target in targetSet:
			for attributeSet in attributes:
				targets_eval.append(s_function_for_t_word(
				    dict_vectors, target, attributeSet))
	m_score = np.mean(targets_eval)
	return m_score, targets_eval

#############################
# BIAS BY NEIGHBORHOOD
#############################


def calculate_bias_by_clustering(model_original, model_debiased, biased_words, topn):

    k_neighbors = finding_neighbors_before_after(
        biased_words, model_original, model_debiased, topn=topn)

    scores_before, scores_after = [], []

    for word in biased_words:

        neighbors_biased_before = len(
            [w for w in k_neighbors[word]["before"] if w in biased_words])
        neighbors_biased_after = len(
            [w for w in k_neighbors[word]["after"] if w in biased_words])

        scores_before.append(neighbors_biased_before)
        scores_after.append(neighbors_biased_after)
    print("avg. number of biased neighbors before: {}; after: {}".format(
        np.mean(scores_before), np.mean(scores_after)))
