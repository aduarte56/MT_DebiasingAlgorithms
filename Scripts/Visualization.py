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


RANDOM_STATE = 42
TSNE_RANDOM_STATE=42

def plot_bias_bar(df_long, plot_title,words_title): 
  #takes a DF with 3 columns: index, value, variable that corresponf to occupations, gender bias score, and embedding
  fig=px.bar(df_long, x="value",y="index", color="variable", orientation="h", barmode="group",
           labels=dict(title=plot_title, index=words_title, value="Gender Bias", variable="Embeddings"),
           height=1000, width=800) 
  fig.update_xaxes(range=[-0.5,0.5])
  #fig.write_image("dataDoubleHard/barplot.png")
  return fig.show()


def plot_bias_bar_direct_bias(df_long, plot_title, words_title):
  #takes a DF with 3 columns: index, value, variable that corresponf to occupations, gender bias score, and embedding
  fig = px.bar(df_long, x="value", y="index", color="variable", orientation="h", barmode="group",
               labels=dict(title=plot_title, index=words_title,
                           #index="Wino Gender Occupations",
                           value="Gender Bias", variable="Embeddings"),
               height=1000, width=800)
  fig.update_xaxes(range=[0, 0.5], dtick=0.05)
  #fig.write_image("dataDoubleHard/barplot.png")
  return fig.show()

def visualize(vectors, y_true, y_pred, ax, title):
    # perform TSNE
    vectors =utils.normalize(vectors)
    X_embedded = TSNE(n_components=2, random_state=RANDOM_STATE).fit_transform(vectors)
    for x,p,y in zip(X_embedded, y_pred, y_true):
        if y:
            ax.scatter(x[0], x[1], marker = '.', c = 'c')
        else:
            ax.scatter(x[0], x[1], marker = 'x', c = 'darkviolet')
    
    return ax

def cluster_and_visualize(words, X1, title, y_true, num=2):
    
    kmeans_1 = KMeans(n_clusters=num, random_state=RANDOM_STATE).fit(X1)
    y_pred_1 = kmeans_1.predict(X1)
    correct = [1 if item1 == item2 else 0 for (item1,item2) in zip(y_true, y_pred_1) ]
    #print('precision', max(sum(correct)/float(len(correct)), 1 - sum(correct)/float(len(correct))))
    print('precision', sum(correct)/float(len(correct)))
    fig, axs = plt.subplots(1, 1, figsize=(6, 3))
    ax1 = visualize(X1, y_true, y_pred_1, axs, title)


def tsne(vecs, labels, title="", ind2label = None, words = None, metric = "l2"):

  tsne = TSNE(n_components=2)#, angle = 0.5, perplexity = 20)
  vecs_2d = tsne.fit_transform(vecs)
  label_names = sorted(list(set(labels.tolist())))
  num_labels = len(label_names)

  names = sorted(set(labels.tolist()))

  plt.figure(figsize=(6, 5))
  colors = "red", "blue"
  for i, c, label in zip(sorted(set(labels.tolist())), colors, names):
     plt.scatter(vecs_2d[labels == i, 0], vecs_2d[labels == i, 1], c=c,
                label=label if ind2label is None else ind2label[label], alpha = 0.3, marker = "s" if i==0 else "o")
     plt.legend(loc = "upper right")

  plt.title(title)
  #plt.savefig("embeddings.{}.png".format(title), dpi=600)
  plt.show()
  return vecs_2d

def perform_purity_test(vecs, k, labels_true):
        np.random.seed(0)
        clustering = KMeans(n_clusters = k)
        clustering.fit(vecs)
        labels_pred = clustering.labels_
        score = sklearn.metrics.homogeneity_score(labels_true, labels_pred)
        return score

def compute_v_measure(vecs, labels_true, k=2):
    
        np.random.seed(0)
        clustering = KMeans(n_clusters = k)
        clustering.fit(vecs)
        labels_pred = clustering.labels_
        return sklearn.metrics.v_measure_score(labels_true, labels_pred)
    


import matplotlib.pyplot as plt
import matplotlib.cm as cm
#plots the t-SNE visualization of neighbor clusters to chosen words
def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(10, 6))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color.reshape(1, -1), alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=6)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
       plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()