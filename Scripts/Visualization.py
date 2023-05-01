"""
@author: angeladuartepardo

This script contains the functions to plot my results. Contains adaptations of the clustering algorithms employed by Gonen et al., 
Boxlots for analyzing the bias scores, neighbor frequencies, etc,. And functions to plot clusters of neighbors with tSNE. 

This file can also be imported as a module and contains the following
functions:
    * plot_bias_bar - to plot the barplot of the bias scores in the original and debiased embeddings
    * plot_bias_bar_direct_bias - to plot the barplot of the direct bias scores
    * plot_top_biased_words - to plot a bar plot of the top 20 most biased words with all the scores of the three methods
    * visualize - class to load and clean the embeddings 
    * cluster_and_visualize - to cluster the neighbors of a word and plot them with tSNE
    * tsne_plot_similar_words - to plot the neighbors of a word with tSNE
    * plot_frequency_original_neighbors   - to plot the frequency of the neighbors of a word in the original embeddings
    * plot_average_distance - to plot the average distance of the original neighbors to a list of words pre- and post-debiasing
"""
import plotly.graph_objects as go
import plotly_express as px
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import sklearn
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import Scripts.utils as utils
import numpy as np


#######################################################################
## BAR PLOTS BIAS SCORES
########################################################################

def plot_bias_bar(df_long, plot_title,words_title): 
  """"
  Funtion to plot the barplot of the bias scores in the original and debiased embeddings
  ----
  :param df_long: dataframe with the bias scores
  :param plot_title: title of the plot
  :param words_title: title of the words
  :return: plot
  """
  #takes a DF with 3 columns: index, value, variable that corresponf to occupations, gender bias score, and embedding
  fig=px.bar(df_long, x="value",y="index", color="variable", orientation="h", barmode="group",
           labels=dict(title=plot_title, index=words_title, value="Gender Bias", variable="Embeddings"),
           height=1000, width=800,
           template='ggplot2',
             color_discrete_sequence=["rgb(246,0,0)", "rgb(0,118,101)"]
           ) 
  fig.update_xaxes(range=[-0.5,0.5])
  #fig.write_image("dataDoubleHard/barplot.png")
  return fig.show()


def plot_bias_bar_direct_bias(df_long, plot_title, words_title):
  """"
  Funtion to plot the barplot of the direct bias scores
  ----
  :param df_long: dataframe with the bias scores
  :param plot_title: title of the plot
  :param words_title: title of the words
  :return: plot
  """
  #takes a DF with 3 columns: index, value, variable that corresponf to occupations, gender bias score, and embedding
  fig = px.bar(df_long, x="value", y="index", color="variable", orientation="h", barmode="group",
               labels=dict(title=plot_title, index=words_title,
                           #index="Wino Gender Occupations",
                           value="Gender Bias", variable="Embeddings"),
               height=1000, width=800)
  fig.update_xaxes(range=[0, 0.5], dtick=0.05)
  #fig.write_image("dataDoubleHard/barplot.png")
  return fig.show()


#plot a bar plot of the top 20 most biased words with all the scores of the three methods


def plot_top_biased_words(df, plt_title, n_words=20):
    """"
    Function to plot the top n_words most biased words
    ----
    :param df: dataframe with the bias scores
    :param plt_title: title of the plot
    :param n_words: number of words to plot
    :return: plot
    """
    df_top = df.head(n_words)
    #remove the simple_bias_score column
    #df_top = df_top.drop(columns=['simple_bias_score'])
    df_top = df_top.reset_index()
    df_top = df_top.rename(columns={'index': 'word'})
    df_top = df_top.melt(
        id_vars=['word'], var_name='score_type', value_name='score')
    fig = px.bar(df_top, x="score", y="word",
                 color="score_type", barmode="group", orientation='h',
                 height=1000, width=800, title=plt_title)

    fig.show()

#######################################################################
## VISUALIZE WORD CLUSTERS
########################################################################

def visualize(vectors, y_true, ax, title, random_state):
    """"
    Function to visualize the masculine and feminine clusters following Gonen et al. (2019)
    ----
    :param vectors: vectors of the feminine and masculine words to be plotted
    :param y_true: true labels
    :param y_pred: predicted labels
    :param ax: axis
    :param title: title of the plot
    :return: plot
    """
    # perform TSNE
    #vectors =utils.normalize(vectors)
    X_embedded = TSNE(
        n_components=2, random_state=random_state).fit_transform(vectors)
    for x,y in zip(X_embedded, y_true):
        if y:
            ax.scatter(x[0], x[1], marker = '.', c = 'c')
        else:
            ax.scatter(x[0], x[1], marker = 'x', c = 'darkviolet')
    ax.set_title(title)
    return ax

def cluster_and_visualize(words, X, title, y_true, random_state,num=2):
    """"
    Function to cluster the words depending on their label. Following Gonen et al. (2019)
    ----
    :param words: words to be plotted
    :param X1: vectors to be plotted
    :param title: title of the plot
    :param y_true: true labels
    :param num: number of clusters
    :return: plot
    """
    
    kmeans= KMeans(n_clusters=num, random_state=random_state, n_init=10).fit(X)
    y_pred = kmeans.predict(X)
    #gets a list with the words that were correctly clustered
    correct = [1 if item1 == item2 else 0 for (item1,item2) in zip(y_true, y_pred) ]
    #A precision score of 1.0 indicates perfect clustering
    precision = sum(correct)/float(len(correct))
    print('precision', precision)
    #creates the plot
    _, axs = plt.subplots(1, 1, figsize=(6, 3))
    ax1 = visualize(X, y_true, axs, title, random_state)
    return precision, ax1


############################################
## PLOTS NEIGHBORS
############################################

import matplotlib.pyplot as plt
import matplotlib.cm as cm
#plots the t-SNE visualization of neighbor clusters to chosen words
def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    """"
    Function to plot the clusters of neighbors using TSNE 
    ----
    :param title: title of the plot
    :param labels: labels of the vectors
    :param embedding_clusters: embedding clusters
    :param word_clusters: word clusters
    :param a: alpha
    :param filename: filename
    :return: plot
    """
    plt.figure(figsize=(14,10 ))
    #because there are many points, I'll use rainboow colors to plot them
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        #using scatter to plot the points
        plt.scatter(x, y, c=color.reshape(1, -1), alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=6)
    
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.title(title)
    plt.grid(True)
    if filename:
       plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_frequency_original_neighbors(df_freq, title_plot, xlabel):
    """
    Function to plot the frequency of the original neighbors in the debiased embeddings
    ----
    :param df_freq: dataframe with the frequency of the original neighbors
    :param title_plot: title of the plot
    :param xlabel: label of the x axis
    :return: plot
    """
    fig = px.bar(df_freq, x='word', y='freq', title='Proportion of original 50 neighbours in the debiased k-vicinity of each word',
                 labels=dict(title=title_plot, freq='Proportion', word=xlabel),
                 height=500, width=1000, 
                 template='ggplot2',
                 color_discrete_sequence=['#424249']*len(df_freq)
                 )
    #update the layout of the plot: update the y axis to be between 0 and 0.5 and the x axis to include all ticks
    fig.update_layout(yaxis=dict(range=[0, 1]))

    fig.show()


#plot the average distance to neighbors before and after debiasing
def plot_average_distance(df_average, title_plot, xlabel):
    """
    Function to plot the average distance to neighbors before and after debiasing
    ----
    :param df_average: dataframe with the average distance to neighbors before and after debiasing
    :param title_plot: title of the plot
    :param xlabel: label of the x axis
    :return: plot
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_average.word,
                         y=df_average['distance_original'], name='Original Embeddings',
                         marker_color='#b2868e'))
    fig.add_trace(go.Bar(x=df_average.word,
                         y=df_average['distance_debiased'], name='Debiased Embeddings',
                         marker_color='#046f94'))
    #add title  to the plot
    fig.update_layout(
        title_text=title_plot, template='ggplot2', barmode='group')
    #change x axis title
    fig.update_xaxes(title_text=xlabel)
    #change y axis title
    fig.update_yaxes(title_text='Average Cosine Distance to Neighbors')
    #update heigh and width
    fig.update_layout(height=500, width=1200)
    fig.show()
