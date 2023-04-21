from typing import Dict, List, Tuple, Union
import numpy as np
import tqdm
from tqdm import tqdm
import string 
import operator

#Downloading the glove embeddings from gensim
import gensim.downloader
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
#import torch

#Class to load and clean the embeddings
class Embeddings(object):
  
  def __init__(self, file, gensim=True):
      """"
      Constructor of the class
      ----
        :param file: path to the file with the embeddings
        :param gensim: True if the embeddings are to be downloaded directly from gensim, False if they are to be read from the local file into a gensim object
      """
      
      if file is None:
            print("Invalid file")
      
      if gensim:
            self.file = file
            print("Loading", self.file, "embeddings")
            self.model=self.load_vectors(file)
            self.vectors, self.words, self.word2idx=self.get_words_vectors(self.model)
      else:
            self.file = file
            print("Loading", self.file, "embeddings")
            self.model = KeyedVectors.load_word2vec_format(file, binary=False)
            self.vectors, self.words, self.word2idx=self.get_words_vectors(self.model)
   

  def load_vectors(self, file):
      """"
        Loads the embeddings from the file directly from gensim
        ----
        :param file: path to the file with the embeddings
      """
      model = gensim.downloader.load(file) #getting the desired model. 
      return model

  def get_words_vectors(self, model):
      """"
        Gets the words and vectors from the model
        ----
        :param file: model object from which the words and vectors are to be extracted
      """
      vectors = model.vectors #list of arrays or vectos
      words = model.index_to_key #list of words
      word2idx = {word: idx for idx, word in enumerate(words)} #dictionary: word, index
      #weights = torch.FloatTensor(model.vectors) #vectors as tensors
      print(f"vectors shape: {vectors.shape}, word2idx length: {len(word2idx)}, vocab length: {len(words)}")
      return vectors, words, word2idx

  def get_word_vector_dict(self):
      """"
      Gets a dictionry with the words as keys and the vectors as values
      ----
      :param file: model object from which the words and vectors are to be extracted
      :return: dictionary with the words as keys and the vectors as values
      """
      #creating a dictionary to access the embeddings with word as key and vectors as values
      dict_vectors = dict({})
      for idx, key in enumerate(self.model.key_to_index):
          dict_vectors[key] = self.model[key]
      return dict_vectors

  def get_vectors_from_list(self, list_words):
    """"
    Gets the vectors of the words in the list
    ----
    :param list_words: list of words
    :return: list of vectors
    """
    
    #Sale de null-it-out
    vectors = []
    for word in list_words:  
        vectors.append(self.model[word])
    return np.array(vectors)
  
  def extract_vectors(self, words, w2i):
    """"
    Extracts the vectors of the words
    ----
    :param words: list of words
    :param w2i: dictionary with the words as keys and the indices as values
    :return: list of vectors
    """
    
    X = [self.vectors[w2i[x],:] for x in words]
    
    return X

  def has_punct(self, word):
    """"
    Checks if the word contains punctuation
    ----
    :param word: word
    :return: True if the word contains punctuation, False otherwise
    """
    if any([punct in string.punctuation for punct in word]):
        return True
    return False

  def has_digit(self,word):
    """"
    Checks if the word contains digits
    ----
    :param word: word
    :return: True if the word contains digits, False otherwise
    """
    if any([digit in string.digits for digit in word]):
        return True
    return False

  def exclude_punctuation(self, words):
    """"
    Excludes the words that contain punctuation or digits
    ----
    :param words: list of words
    :return: list of words without punctuation
    """
    vocab_limited = []
    for word in tqdm(words[:len(words)]): 
        if word.lower() != word :
            continue
        if len(word) >= 20: 
            continue
        if self.has_digit(word) | self.has_punct(word): 
            continue
        if '_' in word:
            p = [self.has_punct(subw) for subw in word.split('_')]
            if not any(p):
                vocab_limited.append(word)
            continue
        vocab_limited.append(word)
    return vocab_limited

  def limit_vocab(self, word_vector, word_index, vocab, exclude = None, exclude_punct = True):
    """"
    Limits the vocabulary to the words that are not in the exclude list
    ----
    :param word_vector: word vectors
    :param word_index: dictionary with the words as keys and the indices as values
    :param vocab: list of words
    :param exclude: list of words to be excluded
    :return: limited vocabulary, limited word vectors, limited word index, limited dictionary with the words as keys and the vectors as values
    """
    vocab_limited=vocab
    if exclude_punct:
      vocab_limited=self.exclude_punctuation(vocab)
    if exclude:
       vocab_limited = list(set(vocab_limited) - set(exclude))

    print("Size of limited vocabulary:", len(vocab_limited))
    
    wv_limited = np.zeros((len(vocab_limited), len(word_vector[0, :])))
    for index,word in enumerate(vocab_limited):
        wv_limited[index,:] = word_vector[word_index[word],:]
    
    w2i_limited = {word: index for index, word in enumerate(vocab_limited)}
    dict_vectors_limit = {word: wv_limited[index,:] for index, word in enumerate(vocab_limited)}
    return vocab_limited, wv_limited, w2i_limited, dict_vectors_limit

  def save_in_word2vec_format(self, vecs: np.ndarray, words: np.ndarray, fname: str):
    """
    Saves the vectors in the word2vec format.
    :param vecs: vectors
    :param words: vocabulary
    :param fname: path to the file where the vectors are to be saved
    """
    with open(fname, "w", encoding = "utf-8") as f:
        f.write(str(len(vecs)) + " " + str(vecs.shape[1]) + "\n")
        for i, (v,w) in tqdm(enumerate(zip(vecs, words))):
            vec_as_str = " ".join([str(x) for x in v])
            f.write(w + " " + vec_as_str + "\n")


#Function that loads the vectors into a KeyedVectors object from the Gensim Package (for debiased embeddings)
def load_word_vectors(fname):
    """
    Loads word vectors from a file.
    :param fname: path to the file with the embeddings
    :return: model:KeyedVectors object, vecs: vectors, words: vocabulary
    """
    model = KeyedVectors.load_word2vec_format(fname, binary=False)
    vecs = model.vectors
    words = list(model.index_to_key)
    return model, vecs, words

#Function that creates a KeyedVectors object from the vectors and the vocabulary
def create_KeyedVectors(vectors, vocab, dimensions):
    """"
    Creates a KeyedVectors object from the vectors and the vocabulary
    ----
    :param vectors: vectors
    :param vocab: vocabulary
    :param dimensions: dimensions of the vectors
    :return: KeyedVectors object
    """
    kv = KeyedVectors(dimensions)
    kv.add_vectors(vocab, vectors)
    return kv

# get a dictionary with the debiased vectors as values and the words as keys, using debiased_vectors, debiased_vocab, debiased_word2idx from hard-debias function.
def get_debiased_dict(wv_debiased, w2i_partial):
   """"
   Gets a dictionary with the debiased vectors as values and the words as keys
   ----
   :param wv_debiased: debiased vectors
   :param w2i_partial: dictionary with the words as keys and the indices as values
   :return: dictionary with the debiased vectors as values and the words as keys
   """
   debiased_dict = {}
   for word, index in w2i_partial.items():
      debiased_dict[word] = wv_debiased[index, :]
   return debiased_dict
