"""
@author: angeladuartepardo

This script contains the functions used to evaluate the debiasing procedure and its consequences on the debiasing space.
I follow the approaches of various authors that are cited along the code. 

This file can also be imported as a module and contains functions for the following evaluations:
    * Pre/post bias scores - functions to compare pre and post bias scores
    * random words test - functions to analyze the vicinities of random words
    * Bias by neighbor - functions to analyze the bias of a word based on their biased neighbors. Follow the approach of Gonen et al. 2019
    * Two Sample Permutation Test - functions to perform the two sample permutation test
    * WEFAT - functions to perform the WEFAT test by Caliskan et al. 2017
"""


from sklearn.manifold import TSNE
from Scripts.utils import *
import itertools
import scipy.misc as misc
import scipy
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
from Scripts.utils import cosine_similarity
import pandas as pd



#############################
# PRE/POST BIAS SCORES
#############################

#compute the bias direction of a word through cosine similarity to the bias direction
def compute_similarity_to_bias_direction(dict_vec_cleaned, bias_direction):
    """"
    Compute the similarity of each word in the dictionary to the bias direction
    :param dict_vec_cleaned: dictionary of word vectors
    :param bias_direction: bias direction
    :return: dictionary of words and their similarity to the bias direction
    """
    #bias_direction = bias_direction / np.linalg.norm(bias_direction)
    similarity = {}
    for word in dict_vec_cleaned.keys():
    #    dict_vec_cleaned[word] = dict_vec_cleaned[word] / np.linalg.norm(dict_vec_cleaned[word])
        similarity[word] = cosine_similarity(bias_direction, dict_vec_cleaned[word]).astype(float)
    return similarity



# Function to compute the gender bias of a word.
# Outputs a dictionary with words as keys and gender bias as values
def compute_gender_simple_bias(dict_vectors, he_embedding, she_embedding):
    """"
    Funtion to compute gender bias as the difference between the cosine similarity of a word to the he embedding and the cosine similarity of the word to the she embedding
    :param dict_vectors: dictionary of word vectors
    :param he_embedding: embedding of the word "he"
    :param she_embedding: embedding of the word "she"
    :return: dictionary of words and their bias
    """
    gender_bias = {}
    for word in dict_vectors.keys():
        vector = dict_vectors[word]
        gender_bias[word] = cosine_similarity(
            vector, she_embedding) - cosine_similarity(vector, he_embedding)
    return gender_bias


def compute_direct_bias(dict_vectors, word_list, bias_subspace):
    """"
    Funtion to compute the direct bias of a word as the cosine similarity of the word to the bias subspace
    :param dict_vectors: dictionary of word vectors
    :param word_list: list of words to compute the bias for
    :param bias_subspace: bias subspace
    :return: dictionary of words and their direct bias
    """
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
    """"
    Funtion to compute the average direct bias of a list of words
    :param dict_vectors: dictionary of word vectors
    :param words_list: list of words to compute the bias for
    :param bias_subspace: bias subspace
    :return: average bias
    """
    directBias = compute_direct_bias(dict_vectors, words_list, bias_subspace)
    #get the norm of the bias for each word first
    for word in directBias.keys():
        directBias[word] = np.linalg.norm(directBias[word])
    #then compute the average
    average_bias = np.mean(list(directBias.values()))
    return average_bias


def get_bias_score_df_from_list(bias_scores_original, debiased_scores, word_list, vocab_cleaned, debiased_vocab_limited):
    """"
    Function to get a dataframe with the bias scores of a list of words
    :param bias_scores_original: dictionary of bias scores before debiasing
    :param debiased_scores: dictionary of bias scores after debiasing
    :param word_list: list of words to compute the bias for
    :param vocab_cleaned: list of words in the vocabulary
    :param debiased_vocab_limited: list of words in the debiased vocabulary
    :return: dataframe with the bias scores of a list of words
    """
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
# RANDOM WORDS TEST
#############################
from collections import defaultdict
#Getting the similar words the gensim way for sanity checks
#the function uses gensim .most_similar() method to find the top-N most similar words to a given word
def finding_neighbors_before_after(word_list, original_model, debiased_model, topn=3):
    """"
    Function to find the top-N most similar words to a given word before and after debiasing (the gensim way)
    :param word_list: list of words to get the neighbors from
    :param original_model: original model
    :param debiased_model: debiased model
    :param topn: number of neighbors to find
    :return: dictionary of words and their neighbors before and after debiasing
    """
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

#Get the embeddings of the neighbors from the words in the word_list for plotting!
def get_embeddings_neighbors(keys,original_model, model_cleaned, topn):
    """"
    Function to get the embeddings of the neighbors from the words in the word_list (the gensim way)
    :param keys: list of words  to get the neighbors from
    :param original_model: original model
    :param model_cleaned: debiased model
    :param topn: number of neighbors to find
    :return: embedding clusters, debiased embedding clusters, word clusters
    """
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


def get_vectors_for_tsne(keys, model_original, model_debiased, k=50):
    """"
    Function to prepare for plotting the embeddings of the neighbors from the words in the word_list! (the gensim way)
    :param keys: list of words  to get the neighbors from
    :param model_original: original model
    :param model_debiased: debiased model
    :param k: number of neighbors to find
    :return: embeddings_en_2d, db_embeddings_en_2d, word_clusters
    """
    embedding_clusters, db_embedding_clusters, word_clusters = get_embeddings_neighbors(
        keys, model_original, model_debiased, k)

    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=2, n_components=2,
                            init='pca', n_iter=3500, random_state=42)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(
        embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    db_embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(
        db_embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    return embeddings_en_2d, db_embeddings_en_2d, word_clusters

#Function to find the top-k most similar words to a given word using the cosine similarity
def get_topK_neighbors(word, dict_vect, vocab, vectors, w2i, k=10):
    """"
    Function to find the top-k most similar words to a given word using the cosine similarity (from scratch, inspired by Gonen et al. 2019)
    :param word: word to compute the neighbors from
    :param dict_vect: dictionary of words and their embeddings
    :param vocab: list of words in the vocabulary
    :param vectors: array of embeddings
    :param w2i: dictionary of words and their indices
    :param k: number of neighbors to find
    :return: dictionary of words and their neighbors before and after debiasing
    """
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
    """"
    Function to find the top-k most similar words to a list of words using the cosine similarity (from scratch, inspired by Gonen et al. 2019)
    :param list_words: list of words to compute the bias for
    :param dict_vect: dictionary of words and their embeddings
    :param vocab: list of words in the vocabulary
    :param vectors: list of embeddings
    :param w2i: dictionary of words and their indices
    :param k: number of neighbors to find
    :return: dictionary of words and their neighbors before and after debiasing
    """
    k_neigh ={}
    for w in tqdm(list_words):
        dict_neigh, _ = get_topK_neighbors(
            w, dict_vect, vocab, vectors, w2i, k)
        k_neigh.update(dict_neigh)
    return k_neigh

#get a list of the neighbors for each word of the dictionary k_neigh
def get_list_neighbors(k_neigh):
    """"
    Function to get a list of the neighbors for each word of the dictionary k_neigh
    :param k_neigh: dictionary of words and their neighbors
    :return: list of the neighbors for each word of the dictionary k_neigh
    """
    list_neigh = []
    for w in k_neigh.keys():
        list_neigh.append([i[0] for i in k_neigh[w]])
    return list_neigh

#function to get the frequency of the original neighbors among the 50 nearest neighbors of selected words
def get_frequency_original_neighbors(list_words, dict_neigh, dict_vect_debiased, vocab_debiased, vectors_debiased, w2i_debiased, neighbours_num=50):
    """"
    Function to get the frequency of the original neighbors among the 50 nearest neighbors of selected words
    :param list_words: list of words to compute the bias for
    :param dict_neigh: dictionary with the random words as keys and the list of tuples of their neighbors and their similarity 
    :param dict_vect_debiased: dictionary of words and their embeddings
    :param vocab_debiased: list of words in the vocabulary
    :param vectors_debiased: list of embeddings
    :param w2i_debiased: dictionary of words and their indices
    :param neighbours_num: number of neighbors to find
    :return: list of the frequency of the original neighbors among the 50 nearest neighbors of selected words
    """
    scores = []
    for idx, word in tqdm(enumerate(list_words)):
        #get the top 50 neighbors of the word
        _, top = get_topK_neighbors(word, dict_vect_debiased, vocab_debiased, vectors_debiased, w2i_debiased,
                                    k=neighbours_num)

        count = 0
        #check if the original neighbors are in the top 50
        for t in top:
            org_neighs = [neigh_tuple[0] for neigh_tuple in dict_neigh[word]]

            if t in org_neighs:
                #print(t)
                count += 1

        scores.append([word, count, count/neighbours_num])
        #print(top)
    return scores


#getting average cosine distance to neighbors before and after debiasing
def get_distance_to_neighbors(k_neigh_original, dict_vectors, debiased_dict):
    """"
    Function to get the average cosine distance to neighbors before and after debiasing
    :param k_neigh_original: dictionary or original neighbors
    :param dict_vectors: dictionary of words and their embeddings
    :param debiased_dict: dictionary of words and their debiased embeddings
    :return: average cosine distance to neighbors before and after debiasing
    """
    distances_original = {}
    distances_debiased = {}
    #loop through the random words
    for r_word in k_neigh_original.keys():
        dist1 = []
        dist2 = []
        #loop through the neighbors of the word
        for list in k_neigh_original[r_word]:
            #add the word and its cosine distance in the original embeddings to the list
            neigh = list[0]
            dist1.append(
                [neigh, 1-cosine_similarity(dict_vectors[r_word], dict_vectors[neigh])])
            #add the word and its cosine distance in the debiased embeddings to the list
            dist2.append(
                [neigh, 1-cosine_similarity(debiased_dict[r_word], debiased_dict[neigh])])
        #add the list of distances to the dictionary
        distances_original[r_word] = dist1
        distances_debiased[r_word] = dist2
    return distances_original, distances_debiased

#get dataframe of distances from distances_original and distances_debiased
def get_df_distances(distances_original,distances_debiased):
    """"
    Function to get a dataframe of distances from distances_original and distances_debiased
    :param distances_original: dictionary of words and their neighbors and cosine distance to them in the original embeddings
    :param distances_debiased: dictionary of words and their neighbors and cosine distance to them in the debiased embeddings
    :return: dataframe of distances from distances_original and distances_debiased
    """
    df=pd.DataFrame()
    for word in distances_original.keys():
        for i in range(len(distances_original[word])):
            #df=df.append({'word':word, 'neighbor':distances_original[word][i][0], 'distance_original':distances_original[word][i][1], 'distance_debiased':distances_debiased[word][i][1]}, ignore_index=True)
            df=pd.concat([df, pd.DataFrame.from_records([{'word':word, 'neighbor':distances_original[word][i][0], 'distance_original':distances_original[word][i][1], 'distance_debiased':distances_debiased[word][i][1]}])], ignore_index=True)
            
    return df


#create a function to run the whole process of getting the neighbors of the original vectors and the debiased vectors and then getting the average distance to neighbors before and after debiasing for random words
# return a dataframe with the average distance to neighbors before and after debiasing for each iteration
def get_df_random_words_neighbor_analysis(random_words,vocab_cleaned, dict_vects,vectors_cleaned, word2idx_cleaned, deb_dict, deb_vocab, deb_vect, deb_word2idx, k=2):
    """"
    Function to run the whole process of getting the neighbors of the original vectors and the debiased vectors and then getting the average distance to neighbors before and after debiasing for random words
    :param random_words: list of random words to compute neighbors from
    :param vocab_cleaned: list of words in the vocabulary
    :param dict_vects: dictionary of words and their embeddings
    :param vectors_cleaned: list of embeddings
    :param word2idx_cleaned: dictionary of words and their indices
    :param deb_dict: dictionary of words and their debiased embeddings
    :param deb_vocab: list of words in the debiased vocabulary
    :param deb_vect: list of debiased embeddings
    :param deb_word2idx: dictionary of words and their indices in the debiased vocabulary
    :param k: number of neighbors to find
    :param size_random_set: number of random words to select
    :return: dataframe with the frequencies of previous neighbors and average distance to neighbors before and after debiasing for each iteration
    """

    #get the neighbors of the original vectors
    k_neigh_original = get_k_nearest_neighbors(
        random_words, dict_vects, vocab_cleaned, vectors_cleaned, word2idx_cleaned, k=k)

    #get the distances to neighbors before and after debiasing
    distances_original, distances_debiased = get_distance_to_neighbors(k_neigh_original,
                                                                       dict_vects, deb_dict)
    #get dataframe of distances from distances_original and distances_debiased
    df_neigh_distances = get_df_distances(
        distances_original, distances_debiased)
    #use df_neigh_distances to get the average distance original and distance debiased for each word using pandas
    df_average = df_neigh_distances[[
        'word', 'distance_original', 'distance_debiased']].groupby('word').mean()

    #frequencies of neighbors
    neig_freq2 = get_frequency_original_neighbors(
        random_words, k_neigh_original, deb_dict, deb_vocab, deb_vect, deb_word2idx, neighbours_num=k)
    df2 = pd.DataFrame(neig_freq2, columns=[
                       'word', 'previous_neighbours', 'freq'])
    #merge the two dataframes on the word column
    df_merged = df2.merge(df_average, on='word')

    return df_merged

#get function to get the values of the average distance to neighbors before and after debiasing for each iteration
def get_df_random_words_neighbor_analysis_values(list_for_random, vocab, dict_vects,vects,word2idx, deb_dict, deb_vocab, deb_vect, deb_word2idx, k=2, num_iterations=1000, size_random_set=2):
    """"
    Function to get a dataframe with the aggregate values of the neighbor analysis: frequency of previous neighbors and average distance to neighbors before and after debiasing for each iteration
    :param list_for_random: list of words to compute neighbors from
    :param vocab: list of words in the vocabulary
    :param dict_vects: dictionary of words and their embeddings
    :param vects: list of embeddings
    :param word2idx: dictionary of words and their indices
    :param deb_dict: dictionary of words and their debiased embeddings
    :param deb_vocab: list of words in the debiased vocabulary
    :param deb_vect: list of debiased embeddings
    :param deb_word2idx: dictionary of words and their indices in the debiased vocabulary
    :param k: number of neighbors to find
    :param num_iterations: number of iterations to run
    :param size_random_set: number of random words to select
    :return: dataframe with the frequencies of previous neighbors and average distance to neighbors before and after debiasing for each iteration
    """
    grand_df = pd.DataFrame()
    for i in range(num_iterations):
        random_words = np.random.choice(list_for_random, size=size_random_set)
        df1 = get_df_random_words_neighbor_analysis(random_words, vocab, dict_vects, vects, word2idx, deb_dict, deb_vocab,
                                                    deb_vect, deb_word2idx, k=k)
        df1['iteration'] = i
        grand_df = pd.concat([grand_df, df1])
    return grand_df

#############################
# BIAS BY NEIGHBORHOOD. Following Gonen et al.2019
#############################


def calculate_bias_by_clustering(model_original, model_debiased, biased_words, topn):
    """"
    Function to compute the bias by neighborhood following the approach of Gonen et al. 2019
    :param model_original: original embeddings on a Gensim format
    :param model_debiased: debiased embeddings on a Gensim format
    :param biased_words: list of biased words
    :param topn: number of neighbors to consider
    """
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


def bias_by_neighbors(bias_original, bias_in_debiased, word_list, dict_vect, vocab, vectors, w2i, neighbours_num=100):
    """"
    Function to compute the bias by neighborhood following the approach of Gonen et al. 2019
    :param bias_original: original bias dictionary
    :param bias_in_debiased: debiased bias dictionary
    :param word_list: list of words to compute the bias by neighborhood
    :param dict_vect: dictionary of words and their embeddings
    :param vocab: list of words in the vocabulary
    :param vectors: list of embeddings
    :param w2i: dictionary of words and their indices
    :param neighbours_num: number of neighbors to consider
    :returns list of word, original and debiased scores, number of feminine and masculine neighbors
    """

    tuples = []
    for word in tqdm(word_list):

        _, top = get_topK_neighbors(
            word, dict_vect, vocab, vectors, w2i, k=neighbours_num)

        m = 0
        f = 0
        for t in top:
            if bias_original[t] > 0:
                f += 1
            else:
                m += 1

        tuples.append(
            (word, bias_original[word], bias_in_debiased[word], f, m))

    return tuples


#############################
# TWO SAMPLE PERMUTATION TEST FOR NEIGHBOR DISTRIBUTION.
#############################
# This test is used to test whether the distribution of neighbors of a word is significantly different before and after debiasing.
# The test is based on the two sample permutation test for equality of distributions.
# The test is an adaptation from the permutation test proposed by Caliskan et al. (2017) for the WEAT test. 
# The code is an adaptation from the interpretation from Gonen et al. (2019) of the permutation test for the WEAT test.

def similarity(word_dict, word1, word2):
    """"
    Function to compute the cosine similarity between two words
    :param word_dict: dictionary of words and their embeddings
    :param word1: first word
    :param word2: second word
    :returns cosine similarity between word1 and word2
    """

    vec1 = word_dict[word1]
    vec2 = word_dict[word2]

    return cosine_similarity(vec1, vec2)


def s_word(word, original_neighbors, dic_vectors, dict_debiased, all_s_words):
    """"
    Function to compute the average cosine similarity between a word and its neighbors before and after debiasing. Word statistic.
    :param word: word to compute the average cosine similarity
    :param original_neighbors: list of neighbors of the word before debiasing
    :param dic_vectors: dictionary of words and their embeddings
    :param dict_debiased: dictionary of words and their debiased embeddings
    :param all_s_words: dictionary of words and their average cosine similarity before and after debiasing
    :returns average cosine similarity between a word and its neighbors before and after debiasing
    """
    if word in all_s_words:
        return all_s_words[word]

    mean_a = []
    mean_b = []

    for a in original_neighbors:
        mean_a.append(similarity(dic_vectors, word, a))
    for b in original_neighbors:
        mean_b.append(similarity(dict_debiased, word, b))

    mean_a = sum(mean_a)/float(len(mean_a))
    mean_b = sum(mean_b)/float(len(mean_b))
    #print('mean_a:',mean_a)
    #print('mean_b:',mean_b)
    all_s_words[word] = [mean_a, mean_b]

    return all_s_words[word]


def s_group(random_words, original_neighbors, dic_vectors,
            dict_debiased, all_s_words):
    """"
    Function to compute the average cosine similarity between a group of words and their neighbors before and after debiasing. Group statistic.
    :param random_words: list of words to compute the average cosine similarity
    :param original_neighbors: list of neighbors of the word before debiasing
    :param dic_vectors: dictionary of words and their embeddings
    :param dict_debiased: dictionary of words and their debiased embeddings
    :param all_s_words: dictionary of words and their average cosine similarity before and after debiasing
    :returns average cosine similarity between a group of words and their neighbors before and after debiasing
    """

    mean_A = float()
    mean_B = float()

    total = 0

    for x in random_words:
        mean_A += s_word(x, original_neighbors, dic_vectors,
                         dict_debiased, all_s_words)[0]

    for y in random_words:
        mean_B += s_word(y, original_neighbors, dic_vectors,
                         dict_debiased, all_s_words)[1]

    #print('meanA:',mean_A)
    #print('mean_B:', mean_B)

    total = (mean_A-mean_B)/float(len(random_words))
    #print(total)

    return total


def p_value_perm_neighs(random_words, original_neighbors, dic_vectors,
                        dict_debiased):
    """"
    Function to compute the p-value of the two sample permutation test for the distribution of neighbors of a word before and after debiasing.
    :param random_words: list of words to compute the average cosine similarity
    :param original_neighbors: list of neighbors of the word before debiasing
    :param dic_vectors: dictionary of words and their embeddings
    :param dict_debiased: dictionary of words and their debiased embeddings
    :returns p-value of the two sample permutation test for the distribution of neighbors of a word before and after debiasing
    """

    np.random.seed(42)
    length = 50
    differences = []
    all_s_words_original = {}
    s_orig = s_group(random_words, original_neighbors, dic_vectors,
                     dict_debiased, all_s_words_original)
    num_of_samples = min(int(scipy.special.comb(50*2, 50)), int(length**2)*100)
    print('original mean:', s_orig)
    print('num of samples', num_of_samples)
    larger = 0
    for i in tqdm(range(num_of_samples)):
        all_s_words = {}
        Xi = np.random.permutation(original_neighbors)[:length]
        #print(s_group(random_words, Xi, dic_vectors,
        #             dict_debiased, all_s_words))
        if abs(s_group(random_words, Xi, dic_vectors,
                       dict_debiased, all_s_words)) > abs(s_orig):  # absolute value because it is a two sample permutation test
            larger += 1

    return larger/float(num_of_samples)



##############################
# WEFAT
###############################
# The following functions follow the interpretation that Gonen et al. 2019 give to the WEAT test proposed by Caliskan et al. 2017.

# word statistic
def s_word_weat(word_dict, w, A, B, all_s_words):
    """"
    Function to compute the average cosine similarity between a word and two sets of words. Word statistic.
    :param word_dict: dictionary of words and their embeddings
    :param w: word to compute the average cosine similarity
    :param A: list of words
    :param B: list of words
    :param all_s_words: dictionary of words and their average cosine similarity
    :returns average cosine similarity between a word and two sets of words
    """

    if w in all_s_words:
        return all_s_words[w]

    mean_a = []
    mean_b = []

    for a in A:
        mean_a.append(similarity(word_dict, w, a))
    for b in B:
        mean_b.append(similarity(word_dict, w, b))

    mean_a = sum(mean_a)/float(len(mean_a))
    mean_b = sum(mean_b)/float(len(mean_b))

    all_s_words[w] = mean_a - mean_b

    return all_s_words[w]


def s_group_weat(word_dict, X, Y, A, B, all_s_words):
    """" 
    Function to compute the average cosine similarity between a group of words and two sets of words. Group statistic.
    :param word_dict: dictionary of words and their embeddings
    :param X: list of words to compute the average cosine similarity
    :param Y: list of words to compute the average cosine similarity
    :param A: list of words
    :param B: list of words
    :param all_s_words: dictionary of words and their average cosine similarity
    :returns average cosine similarity between a group of words and two sets of words
    """

    total = 0
    for x in X:
        total += s_word_weat(word_dict, x, A, B, all_s_words)
    for y in Y:
        total -= s_word_weat(word_dict, y, A, B, all_s_words)

    return total


def p_value_exhust_weat(word_dict, X, Y, A, B):
    """"
    Function to compute the p-value of the WEAT test. Is exhaustive because it goes through all possible combinations of X and Y.
    :param word_dict: dictionary of words and their embeddings
    :param X: list of words to compute the average cosine similarity
    :param Y: list of words to compute the average cosine similarity
    :param A: list of words
    :param B: list of words
    :returns p-value of the two sample permutation test for the WEAT test
    """

    if len(X) > 20:
        print('might take too long, use sampled version: p_value')
        return

    assert(len(X) == len(Y))

    all_s_words = {}
    s_orig = s_group_weat(word_dict, X, Y, A, B, all_s_words)

    union = set(X+Y)
    subset_size = len(union)/2

    larger = 0
    total = 0
    for subset in tqdm(set(itertools.combinations(union, int(subset_size)))):
        total += 1
        Xi = list(set(subset))
        Yi = list(union - set(subset))
        if s_group_weat(word_dict, Xi, Yi, A, B, all_s_words) > s_orig:
            larger += 1
    print('num of samples', total)
    #print(all_s_words)
    return larger/float(total)


def p_value_sample_weat(word_dict, X, Y, A, B):
    """"
    Function to compute the p-value of the WEAT test. Is sampled because it goes through a sample of all possible combinations of X and Y.
    :param word_dict: dictionary of words and their embeddings
    :param X: list of words to compute the average cosine similarity
    :param Y: list of words to compute the average cosine similarity
    :param A: list of words
    :param B: list of words
    :returns p-value of the two sample permutation test for the WEAT test
    """

    np.random.seed(42)

    all_s_words = {}

    assert(len(X) == len(Y))
    length = len(X)

    s_orig = s_group_weat(word_dict, X, Y, A, B, all_s_words)

    num_of_samples = min(1000000, int(
        scipy.special.comb(length*2, length)*100))
    print('num of samples', num_of_samples)
    larger = 0
    for i in range(num_of_samples):
        permute = np.random.permutation(X+Y)
        Xi = permute[:length]
        Yi = permute[length:]
        if s_group_weat(word_dict, Xi, Yi, A, B, all_s_words) > s_orig:
            larger += 1

    return larger/float(num_of_samples)
