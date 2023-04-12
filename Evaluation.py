from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
from utils import cosine_similarity
import pandas as pd

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
