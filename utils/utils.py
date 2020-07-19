import sys
import time
import gensim
import re
import nltk
import os
import gzip

import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
from matplotlib import offsetbox

class Loaders:
    def load_mnist(path, kind='train'):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        return images, labels
        
class Plotters:
    def tsne_plot(model, number_of_words=100):
        "Creates and TSNE model and plots it"
        labels = []
        tokens = []
        i = 0
        for word in model.vocab:
            tokens.append(model[word])
            labels.append(word)
            i+=1
            if i >= number_of_words:
                break

        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(16, 16)) 
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.show()
        plt.savefig("plots/tsne_.png")
    
    def plot_embeddings_for_datafold(X_dmap, model, number_of_points=100):
        "Creates a model and plots it"
        labels = []
        tokens = []
        i = 0
        for word in model.vocab:
            tokens.append(X_dmap[i])
            labels.append(word)
            i+=1
            if i >= number_of_points:
                break

        x = []
        y = []
        for value in tokens:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(16, 16)) 
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.show()
        plt.savefig("plots/word_datafold.png")
    
    def plot_embedding_fashion_mnist(X, y, images, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        """Scale and visualize the embedding vectors"""
        plt.figure(figsize=[10, 10])
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(
                X[i, 0],
                X[i, 1],
                str(y[i]),
                color=plt.cm.Set1(y[i] / 10.0),
                fontdict={"weight": "bold", "size": 9},
            )

        if hasattr(offsetbox, "AnnotationBbox"):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1.0, 1.0]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-2:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(images[i].reshape((28, 28)), cmap=plt.cm.gray_r), X[i]
                )
                ax.add_artist(imagebox)
        #plt.xticks([]), plt.yticks([])

        if title is not None:
            plt.title(title)
        plt.savefig("plots/mnist_.png")
            
    def plot_embedding_pids(X, y, title="train"):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        """Scale and visualize the embedding vectors"""
        plt.figure(figsize=[10, 10])
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(
                X[i, 0],
                X[i, 1],
                str(y[i]),
                color=plt.cm.Set1(y[i] / 10.0),
                fontdict={"weight": "bold", "size": 9},
            )

        if title is not None:
            plt.title(title)
        plt.savefig(f"plots/pid_2d_{title}.png")
        
    def plot_embedding_pids_3d(X, y, title="datafold"):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        """Scale and visualize the embedding vectors"""
        plt.figure(figsize=[10, 10])
        ax = plt.subplot(111, projection='3d')
        for i in range(X.shape[0]):
            ax.text(
                X[i, 0],
                X[i, 1],
                X[i, 2],
                str(y[i]),
                color=plt.cm.Set1(y[i] / 10.0),
                fontdict={"weight": "bold", "size": 9},
            )

        if title is not None:
            plt.title(title)
        plt.savefig(f"plots/pid_3d_{title}.png")

    def plot_embedding_fashion_mnist_3d(X, y, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        """Scale and visualize the embedding vectors"""
        plt.figure(figsize=[10, 10])
        ax = plt.subplot(111, projection='3d')
        for i in range(X.shape[0]):
            ax.text(
                X[i, 0],
                X[i, 1],
                X[i, 2],
                str(y[i]),
                color=plt.cm.Set1(y[i] / 10.0),
                fontdict={"weight": "bold", "size": 9},
            )

        if title is not None:
            plt.title(title)
        plt.savefig("plots/mnist_3d.png")

    
