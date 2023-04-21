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

#compute the bias direction of a word through cosine similarity to the bias direction
def compute_similarity_to_bias_direction(dict_vec_cleaned, bias_direction):
    """"
    Compute the similarity of each word in the dictionary to the bias direction
    :param dict_vec_cleaned: dictionary of word vectors
    :param bias_direction: bias direction
    :return: dictionary of words and their similarity to the bias direction
    """
    bias_direction = bias_direction / np.linalg.norm(bias_direction)
    similarity = {}
    for word in dict_vec_cleaned.keys():
        dict_vec_cleaned[word] = dict_vec_cleaned[word] / \
            np.linalg.norm(dict_vec_cleaned[word])
        similarity[word] = cosine_similarity(bias_direction, dict_vec_cleaned[word])
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
    :return: dictionary of words and their bias
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
    Funtion to compute the average bias of a list of words
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
# CLUSTERING
#############################

def cluster_and_evaluate(X, random_state, y_true, num=2):
    """"
    function to calculate the alignment of the clusters with the true labels
    :param X: data to cluster
    :param random_state: random state
    :param y_true: true labels
    :param num: number of clusters
    :return: kmeans model, predicted labels, data, alignment
    """
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
    """"
    Function to find the top-N most similar words to a given word before and after debiasing
    :param word_list: list of words to compute the bias for
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

#Get the embeddings of the neighbors from the words in the word_list
def get_embeddings_neighbors(keys,original_model, model_cleaned, topn):
    """"
    Function to get the embeddings of the neighbors from the words in the word_list
    :param keys: list of words to compute the bias for
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


#Function to find the top-k most similar words to a given word using the cosine similarity
def get_topK_neighbors(word, dict_vect, vocab, vectors, w2i, k=10):
    """"
    Function to find the top-k most similar words to a given word using the cosine similarity
    :param word: word to compute the bias for
    :param dict_vect: dictionary of words and their embeddings
    :param vocab: list of words in the vocabulary
    :param vectors: list of embeddings
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
    Function to find the top-k most similar words to a list of words using the cosine similarity
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


def get_frequency_original_neighbors(list_words, list_neigh, dict_vect_debiased, vocab_debiased, vectors_debiased, w2i_debiased, neighbours_num=50):
    """"
    Function to get the frequency of the original neighbors among the 50 nearest neighbors of selected words
    :param list_words: list of words to compute the bias for
    :param list_neigh: list of the neighbors for each word of the dictionary k_neigh
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
            if t in list_neigh[idx]:
                print(t)
                count += 1

        scores.append([word, count, count/neighbours_num])
        #print(top)
    return scores

#Create a function to get the average distance of the neighbors of a word in the debiased embeddings to their possition in the original embeddings
def getting_neighbor_av_distance_debiased(neighbors, dict_vectors, debiased_dict):
    """"
    Function to get the average distance of the neighbors of a word in the debiased embeddings to their possition in the original embeddings
    :param neighbors: list of the neighbors for each word of the dictionary k_neigh
    :param dict_vectors: dictionary of words and their embeddings
    :param debiased_dict: dictionary of words and their debiased embeddings
    :return: average distance of the neighbors of a word in the debiased embeddings to their possition in the original embeddings
    """
    distances = []
    for n in neighbors:
        distances.append(cosine_similarity(debiased_dict[n], dict_vectors[n]))
    return np.mean(distances)


#Create a function to get the average distance of the neighbors of a word from the word itself
def gettng_neighbor_av_distance(word, neighbors, vectors, word2idx):
    """"
    Function to get the average distance of the neighbors of a word from the word itself
    :param word: word to compute the bias for
    :param neighbors: list of the neighbors for each word of the dictionary k_neigh
    :param vectors: list of embeddings
    :param word2idx: dictionary of words and their indices
    :return: average distance of the neighbors of a word from the word itself
    """
    distances = []
    for n in neighbors:
        distances.append(np.linalg.norm(
            vectors[word2idx[word]] - vectors[word2idx[n]]))
    return np.mean(distances)

#get dataframe of distances from distances_original and distances_debiased


def get_df_distances(distances_original, distances_debiased):
    df = pd.DataFrame()
    for word in distances_original.keys():
        for i in range(len(distances_original[word])):
            #df=df.append({'word':word, 'neighbor':distances_original[word][i][0], 'distance_original':distances_original[word][i][1], 'distance_debiased':distances_debiased[word][i][1]}, ignore_index=True)
            df = pd.concat([df, pd.DataFrame.from_records([{'word': word, 'neighbor': distances_original[word][i][0],
                           'distance_original':distances_original[word][i][1], 'distance_debiased':distances_debiased[word][i][1]}])], ignore_index=True)

    return df

#getting average cosine distance to neighbors before and after debiasing
def get_distance_to_neighbors(random_words, list_neigh, dict_vectors, debiased_dict):
    """"
    Function to get the average cosine distance to neighbors before and after debiasing
    :param random_words: list of words to compute the bias for
    :param list_neigh: list of the neighbors for each word of the dictionary k_neigh
    :param dict_vectors: dictionary of words and their embeddings
    :param debiased_dict: dictionary of words and their debiased embeddings
    :return: average cosine distance to neighbors before and after debiasing
    """
    distances_original = {}
    distances_debiased = {}
    #loop through the random words
    for i, word in enumerate(random_words):
        dist1 = []
        dist2 = []
        #loop through the neighbors of the word
        for neigh in list_neigh[i]:
            #add the word and its cosine distance in the original embeddings to the list
            dist1.append(
                [neigh, 1-cosine_similarity(dict_vectors[word], dict_vectors[neigh])])
            #add the word and its cosine distance in the debiased embeddings to the list
            dist2.append(
                [neigh, 1-cosine_similarity(debiased_dict[word], debiased_dict[neigh])])
        #add the list of distances to the dictionary
        distances_original[word] = dist1
        distances_debiased[word] = dist2

    return distances_original, distances_debiased

#############################
# MAC SCORES
#############################

from scipy import spatial
import numpy as np

#Function adapted from by Manzini et al. (2018)
def s_function_for_t_word(dict_vectors, target_word, attributes):
    """"
    Function to compute the mean cosine distances between the target word and the attributes
    :param dict_vectors: dictionary of words and their embeddings
    :param target_word: word to compute the bias for
    :param attributes: list of attributes
    :return: mean cosine distances between the target word and the attributes
    """
    attribute_vectors = np.array([dict_vectors[attribute]
	                               for attribute in attributes])
    target_vector = dict_vectors[target_word]
    cosine_distances = spatial.distance.cdist([target_vector], attribute_vectors, metric='cosine').flatten()
    return cosine_distances.mean()


def multiclass_evaluation_MAC(dict_vectors, targets_list, attributes):
    """"
    Function to compute the MAC score
    :param dict_vectors: dictionary of words and their embeddings
    :param targets_list: list of target words
    :param attributes: list of attributes
    :return: MAC score and evaluation score
    """
    
    targets_eval = []
    for targetSet in targets_list:
        for target in targetSet:
            for attributeSet in attributes:
                targets_eval.append(s_function_for_t_word(dict_vectors, target, attributeSet))
    m_score = np.mean(targets_eval)
    return m_score, targets_eval

#############################
# BIAS BY NEIGHBORHOOD
#############################


def calculate_bias_by_clustering(model_original, model_debiased, biased_words, topn):
    """"
    Function to compute the bias by neighborhood
    :param model_original: original embeddings
    :param model_debiased: debiased embeddings
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
