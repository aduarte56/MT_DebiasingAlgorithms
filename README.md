# On the promises and costs of bias mitigation in static word embeddings: an evaluation of the Hard-Debiasing Algorithm

## Summary

This repository hosts the code and replication material for my research project: "On the promises and costs of bias mitigation in static word embeddings: 
        an evaluation of the Hard-Debiasing Algorithm". This is my thesis for the Master of Data Science for Public Policy at Hertie School, Berlin.

### Abstract

NLP algorithms use a wide range of word embeddings to develop language-related tasks. However, many of these embeddings encode stereotypes and biases that can generate differential results for specific groups on downstream tasks. Geometric debiasing algorithms, in particular, the Hard-Debiasing algorithm, respond to the need for fairness in machine learning models because of the mathematical removal of biases. However, despite their initial promising results, some authors have argued that the debiasing performed by these algorithms is superficial and inconsistent. This project aims to evaluate the algorithm to reflect on the trade-offs that occur when using a technical approach of geometric nature for bias mitigation. In addition to presenting the experiments and results of the evaluation, I will argue in the paper that on top of other problems, debiasing algorithms change the geometry of the word embeddings space by disentangling the original word clusters, generating a loss of context that can potentially affect downstream tasks. Data scientists should be cautious when using debiasing algorithms because of the possibility of introducing noise that could further affect already marginalized populations.

### Contents

The repo hosts: 
- 5 scripts with the functions necessary to perform the evaluation (PreprocessingEmbeddings, utils, Visualization, Evaluation and HardDebias)
- 3 notebooks that walk the interested crowd through the two parts of the design: 
  1. The analysis of the internal workings of the algorithm (through the exploration of the Bias Direction and Changes to the Algorithm based on the combination of parameters)
  2. The analyisis of the transformations of the vector space through the implementation of the debiasing algorithm.

The embeddings used can be dowloaded directly from Gensim using the functions of the Embeddings Class or from the following data [folder](https://www.dropbox.com/scl/fo/8hdjy5i8quw5ydytluff9/h?dl=0&rlkey=bjxtp86c409zb0hpof0bitohl). The folder also contains the lists of words used in the process

## Author

- Ángela Duarte Pardo ([website](https://github.com/aduarte56)), 


