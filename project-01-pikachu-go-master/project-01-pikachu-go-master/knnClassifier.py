#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:00:51 2019

@author: kaiyanghan
"""
import os
import numpy as np 
import matplotlib.pyplot as plt
import AbstractClassifier
from scipy.stats import multivariate_normal
from sklearn import neighbors
from sklearn.model_selection import train_test_split,cross_val_score	

#build knn clasifier
# a=np.load('train_data/easy_set/easy_feat/train_easy_data.npy')
# b=np.load('train_data/easy_set/easy_feat/train_easy_labels.npy')

class KnnClassifier(AbstractClassifier.AbstractClassifier):

    def __init__(self, param_list):
        '''
        Initialize weights and whatever needed variables in this method.
        @param param_list: parameters necessary for classifier:
                           - number of neighbors
                           - Type of weights (uniform, distance)
                           - metric to use
        '''
        self.classifier = neighbors.KNeighborsClassifier(param_list[0],
                                                         weights=param_list[1],
                                                         metric=param_list[2])

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def train(self, x, y, epochs=10, batch_size=16, lr_rate=0.001):
        '''
        Trains the classifier using input data and labels, if supervised is desired.
        @param x: Input Data
        @param y: Labels corresponding to the input data
        @param epochs: Number of epochs to run training
        @param batch_size: Batch sizes for iterations w/in an epoch
        @param lr_rate: Learning rate. May or may not be used in this function
        @returns: return the trained KNN classifier
        '''
        return self.classifier.fit(x, y)


    def predict(self, x) -> list:
        '''
        Predict the results of the classifier on a given set of data
        @param x: Input Data to be classified
        @returns: list of predicted labels (this corresponds to given input data)
        '''
        return self.classifier.predict(x)


    def evaluate(self, x, y):
        '''
        Runs classifier on input data and outputs statistics such as accuracy and error
        @param x: Input data
        @param y: Labels corresponding to input data
        @returns: list containing evaluated accuracy and error
        '''
        return [self.classifier.score(x, y), -1]


def clf_knn(data,labels,n_neighbors):
    x_train = np.array(data)
    diff=[]
    for i in range(0,data.shape[1]):
        diff.append(max(x_train[:,i])-min(x_train[:,i]))
        x_train[:,i]=(x_train[:,i]-min(x_train[:,i]))/diff[i]
    clf=neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(x_train,labels)
    return clf


#calculate cross validation
def crov_knn(data,labels,n):
    X=data
    X=np.array(X)
    diff=[]
    #normalize
    for i in range(0,data.shape[1]):
        diff.append(max(X[:,i])-min(X[:,i]))
        X[:,i]=(X[:,i]-min(X[:,i]))/diff[i]
    labels=np.array(labels)
    cv_scores = []	
    for n_neighbors in range(1,n+1):
        clf = clf_knn(data,labels,n_neighbors)   
        scores = cross_val_score(clf,X,labels.T,cv=10,scoring='accuracy')
        cv_scores.append(scores.mean())
    maxn=list(cv_scores).index(max(cv_scores))+1
    return maxn


def knn_test(data,labels,n):
    data_tr,data_te,labels_tr,labels_te=train_test_split(data,labels,test_size=0.2)
    X=np.array(data_te)
    diff=[]
    for i in range(0,data_te.shape[1]):
        diff.append(max(X[:,i])-min(X[:,i]))
        X[:,i]=(X[:,i]-min(X[:,i]))/diff[i]
    maxn = crov_knn(data_tr,labels_tr,n)   
    clf=clf_knn(data_tr,labels_tr,maxn)
    labels_pre=clf.predict(X)
    accu=sum(labels_pre==labels_te)/len(labels_te)
    return accu

