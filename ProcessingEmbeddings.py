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

class Embeddings(object):
  def __init__(self, file, gensim=True):
      if file is None:
            print("Invalid file")
      else:
            self.file = file
            print("Loading", self.file, "embeddings")
            self.model=self.load_vectors(file)
            self.vectors, self.words, self.word2idx=self.get_words_vectors(self.model)
      

  def load_vectors(self, file):
      model = gensim.downloader.load(file) #getting the desired model. 
      return model

  def get_words_vectors(self, model):
      vectors = model.vectors #list of arrays or vectos
      words = model.index_to_key #list of words
      word2idx = {word: idx for idx, word in enumerate(words)} #dictionary: word, index
      #weights = torch.FloatTensor(model.vectors) #vectors as tensors
      print(f"vectors shape: {vectors.shape}, word2idx length: {len(word2idx)}, vocab length: {len(words)}")
      return vectors, words, word2idx

  def get_word_vector_dict(self):
      #creating a dictionary to access the embeddings with word as key and vectors as values
      dict_vectors = dict({})
      for idx, key in enumerate(self.model.key_to_index):
          dict_vectors[key] = self.model[key]
      return dict_vectors

  def get_vectors_from_list(self, list_words):
      #Sale de null-it-out
      vectors = []
      for word in list_words:  
          vectors.append(self.model[word])
      return np.array(vectors)
  
  def extract_vectors(self, words, w2i):
    
    X = [self.vectors[w2i[x],:] for x in words]
    
    return X

  def normalize_embeddings(self):
    self.vectors /= np.linalg.norm(self.vectors)

  def has_punct(self, word):
    if any([punct in string.punctuation for punct in word]):
        return True
    return False

  def has_digit(self,word):
    if any([digit in string.digits for digit in word]):
        return True
    return False

  def exclude_punctuation(self, words):
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

  def limit_vocab(self, word_vector, word_index, vocab, exclude = None):
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
    with open(fname, "w", encoding = "utf-8") as f:
        f.write(str(len(vecs)) + " " + str(vecs.shape[1]) + "\n")
        for i, (v,w) in tqdm(enumerate(zip(vecs, words))):
            vec_as_str = " ".join([str(x) for x in v])
            f.write(w + " " + vec_as_str + "\n")


