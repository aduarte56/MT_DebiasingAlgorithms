import numpy as np
import utils


def find_gender_direction(word_vectors, word2index_partial, definitional_pairs): 
    gender_directions = list()
    for gender_word_list in [definitional_pairs]:
        gender_directions.append(utils.perform_PCA_pairs(gender_word_list, word_vectors, word2index_partial).components_[0])
    return  gender_directions

def remove_gender_component(vocab_partial, vectors, w2i_partial, gender_directions): 
          #Removes the bias component of words that should be neutral
          debiased_vectors = np.zeros((len(vocab_partial), len(vectors[0, :]))).astype(float)
          for i, w in enumerate(vocab_partial):
              u = vectors[w2i_partial[w], :]
              for gender_direction in gender_directions:
                u = utils.remove_vector_projection(u, gender_direction)
                debiased_vectors[w2i_partial[w], :] = u
          return debiased_vectors

def remove_frequency_features(vocab_partial, word_vectors, word2index, word2index_partial, component_ids): 
          #Using the above function to find the main principal components
          main_pca = utils.get_main_pca_all(word_vectors)
          vectors_mean = np.mean(np.array(word_vectors), axis=0)
          #print(pd.DataFrame(word_vectors).isna().sum())
          pca_main=utils.get_main_pca_all(word_vectors)
          components=[]
          for i in component_ids:
              components.append(pca_main.components_[i])
  
          word_vec_frequency = np.zeros((len(vocab_partial), word_vectors.shape[1])).astype(float)
          for i, word in enumerate(vocab_partial):
              vector = word_vectors[word2index[word],:]
              proj = np.zeros(vector.shape).astype(float)
                  # removes the component of vector in the direction of principal_component
              for principal_component in components:
                  proj += np.dot(np.dot(np.transpose(principal_component), vector), principal_component)
              word_vec_frequency[word2index_partial[word], :] = word_vectors[word2index[word], :] - proj -vectors_mean
                                                                                                                                                                                                                                                                                                                                                   
          return word_vec_frequency

def hard_debias(wv, w2i, w2i_partial, vocab_partial, component_ids, definitional_pairs):
        vectors=wv
        # get rid of frequency features
        vectors=remove_frequency_features(vocab_partial, wv, w2i, w2i_partial, component_ids)
        
        # debias
        gender_directions=find_gender_direction(vectors, w2i_partial, definitional_pairs)

        wv_debiased=remove_gender_component(vocab_partial, vectors, w2i_partial, gender_directions) 
        
        return wv_debiased


def getting_optimal_direction(vectors, word2idx, w2i_partial, vocab_partial, male_words, female_words, y_true, definitional_pairs):
        precisions = []
        for component_id in range(20):
          print(f'Component: {component_id}', end=', ')
          
          wv_debiased = hard_debias(vectors, word2idx, w2i_partial, vocab_partial, component_ids = [component_id], definitional_pairs= definitional_pairs)
          _, _, _, precision = utils.cluster_and_evaluate(male_words + female_words, 
                                utils.extract_vectors(male_words + female_words, wv_debiased, w2i_partial), 1, y_true)
          precisions.append(precision)
          optimal_frequency_direction = precisions.index(min(precisions))
        return precisions, optimal_frequency_direction
