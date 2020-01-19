'''
Fundamentals of ML: Project 01

Team: Pikachu Go
Members:
- Spencer Chang
- Kaiyang Han
- Fuyuan Zhang
- Zhewei Song

Description of 'train.py':


'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import neighbors
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, log_loss, adjusted_rand_score

import torch
torch.backends.cudnn.enable=False

import AbstractClassifier
import KaiyangClassifier
import knnClassifier
import loaddata


def make_parser():
    '''
        Make an argument parser to get the file paths for finding the 'data' and 'labels' in .npy file format
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', action='store_true')
    return parser.parse_args()


def norm_data(feat_x):
    '''
        Normalize all features to a Gaussian(0, 1) random variable
    '''
    feat_x = np.array(feat_x)

    # Data must be arranged as matrix of size (# samples) x (# features)
    mu = feat_x.mean(axis=0)
    sigma = feat_x.std(axis=0)

    feat_x = (feat_x - mu) / sigma
    return feat_x


def kfold_knn(clf, data, labels, folds=5, rand_state=None):
    '''
        Puts a classifier through K-Fold cross validation using the sci-kit learn library.
        @param clf - classifier with a 'fit' method taking as input (X, y) and outputs tensor T of predictions
        @param data - Input data for the cross-validation
        @param labels - True labels for the input data
        @param folds - Number of folds for the data
        @param rand_state - Determined state for the cross-validation, defaults to np.random
    '''
    # cv = StratifiedShuffleSplit(n_splits=folds, test_size=(1/folds), random_state=rand_state)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rand_state)
    scores = cross_validate(clf,
                            data,
                            labels,
                            scoring=('accuracy', 'neg_log_loss'),
                            cv=skf.split(data, labels),
                            return_train_score=True)

    val_acc = scores['test_accuracy']
    val_loss = scores['test_neg_log_loss']
    # val_rand = scores['test_adjusted_rand_score']

    train_acc = scores['train_accuracy']
    train_loss = scores['train_neg_log_loss']
    # train_rand = scores['train_adjusted_rand_score']
    return [val_acc, val_loss], [train_acc, train_loss]


def kfold_mlp(neurons:list, data, labels, epochs=1000, batch_size=16, lr=0.001, debug_idx=0, folds=5, rand_state=None):
    '''
        Puts a classifier through K-Fold cross validation using the sci-kit learn library.
        @param clf - classifier to validate
        @param data - Input data for the cross-validation
        @param labels - True labels for the input data
        @param folds - Number of folds for the data
        @param rand_state - Determined state for the cross-validation, defaults to np.random
    '''

    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rand_state)

    for step, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        print("~~~~~~~~~~ FOLD {} ~~~~~~~~~~".format(step+1))

        mlp_clf = KaiyangClassifier.KaiyangClassifier(data[0].shape[0], neurons)

        train_x, val_x = torch.tensor(data[train_idx]).float(), torch.tensor(data[val_idx]).float()
        train_y, val_y = torch.tensor(labels[train_idx]).long()-1, torch.tensor(labels[val_idx]).long()-1

        train_res, val_res = mlp_clf.train(train_x,
                                           train_y,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           lr_rate=lr,
                                           val_x=val_x,
                                           val_y=val_y,
                                           debug_idx=debug_idx)
        
        train_acc.append(train_res[0])
        train_loss.append(train_res[1])
        val_acc.append(val_res[0])
        val_loss.append(val_res[1])

    return [val_acc, val_loss], [train_acc, train_loss]


def hyper_knn(data,labels,folds,maxn,weight_type,metric_type, k_list:list=None, rand_state=1):
    '''
    Conduct tests to check the MLP hyperparameters
    '''
    val = list()
    tr = list()

    if k_list is None:
        iter = range(1, maxn+1, 2)
    else:
        iter = k_list

    for i in iter:
        valtemp=[]
        trtemp=[]
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=i,
                                             weights=weight_type,
                                             metric=metric_type)
        valtemp,trtemp=kfold_knn(knn_clf, data, labels, folds=folds, rand_state=rand_state)
        val.append(valtemp)
        tr.append(trtemp)
    return val,tr
 

def hyper_mlp(data, labels, epochs, batch_size, lr=0.001, debug_idx=0, folds=5, rand_state=1):
    val= list()
    tr=list()
    rates = [0.01]
    neurons = [[16, 8]]
    # neurons = [[20, 8], [16, 8], [8, 8], [4, 8]]
    # neurons=[[22, 18, 12, 8],[18, 12, 8], [12, 8]]
    # neurons=[[18, 12, 8], [12, 8]]

    idx = 1
    for cfg in neurons:
        valtemp=[]
        trtemp=[]
        for r in rates:

            valtemp, trtemp = kfold_mlp(cfg, data, labels, epochs, batch_size, r, folds=folds, rand_state=rand_state)

            # Plot all Accuracy Values
            plt.figure()
            plt.title('MLP Cfg {} - Training Accuracy Per Fold'.format(idx))
            for step, acc in enumerate(trtemp[0]):
                plt.plot(acc, label='Fold {}'.format(step+1))
            plt.legend()

            plt.figure()
            plt.title('MLP Cfg {} - Validation Accuracy Per Fold'.format(idx))
            for step, acc in enumerate(valtemp[0]):
                plt.plot(acc, label='Fold {}'.format(step+1))
            plt.legend()

            # Plot All loss values
            plt.figure()
            plt.title('MLP Cfg {} - CrossEntropy Loss Per Fold'.format(idx))
            for step, acc in enumerate(trtemp[1]):
                plt.plot(acc, label='Fold {}'.format(step+1))
            plt.legend()

            plt.figure()
            plt.title('MLP Cfg {} - Val CrossEntropy Loss Per Fold'.format(idx))
            for step, acc in enumerate(valtemp[1]):
                plt.plot(acc, label='Fold {}'.format(step+1))
            plt.legend()
            idx += 1

            valtemp = [np.asarray(valtemp[0]).mean(axis=0), np.asarray(valtemp[1]).mean(axis=0)]
            trtemp = [np.asarray(trtemp[0]).mean(axis=0), np.asarray(trtemp[1]).mean(axis=0)]
            
            # Remove association with tensors
            trtemp[1] = np.array([tens.item() for tens in trtemp[1]])
            valtemp[1] = np.array([tens.item() for tens in valtemp[1]])

        val.append(valtemp)
        tr.append(trtemp)
    
    return val,tr


if __name__ == '__main__':
    args = make_parser()

    e_train_data = np.load('train_data/easy_set/easy_feat/train_easy_data.npy')
    e_train_labels = np.load('train_data/easy_set/easy_feat/train_easy_labels.npy')

    h_train_data = np.load('train_data/hard_set/hard_feat/train_hard_data.npy')
    h_train_labels = np.load('train_data/hard_set/hard_feat/train_hard_labels.npy')

    # Change this variable depending on whether or not cross-validation is requested
    validation = args.val

    e_train_data = norm_data(e_train_data)
    h_train_data = norm_data(h_train_data)

    # ------------------------------- MLP HYPERPARAMETERS -------------------------------
    n_mlp_folds = 5
    n_epochs = 200
    learning = 0.01
    neurons = [16, 8]
    batch_sz = 32

    # ------------------------------- KNN HYPERPARAMETERS -------------------------------
    n_knn_folds = 10
    n_neighbors = 31
    weight_type = 'distance'
    metric_type = 'minkowski'
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                             weights=weight_type,
                                             metric=metric_type)

    if not validation:
        print("Training the models...")
        knn_clf.fit(e_train_data, e_train_labels)
        
        mlp_clf = KaiyangClassifier.KaiyangClassifier(h_train_data[0].shape[0], neurons)
        train_x = torch.tensor(h_train_data).float()
        train_y = torch.tensor(h_train_labels).long() - 1
        mlp_clf.train(train_x,
                      train_y,
                      epochs=n_epochs,
                      batch_size=batch_sz)

        # Save models to a .pkl file to be loaded into test.py
        loaddata.save_pkl('KNNclassifier.pkl', knn_clf)
        loaddata.save_pkl('MLPclassifier.pkl', mlp_clf)
    else:
        print("Doing Cross-Validation...")
    # print(e_train_data.shape)
    # print(h_train_data.shape)

        # k_val = [7, 13, 31] # 33, 37, 45, 51]
        # val_metrics, train_metrics = hyper_knn(e_train_data,
        #                                        e_train_labels,
        #                                        10,
        #                                        20,
        #                                        weight_type,
        #                                        metric_type,
        #                                        k_list=k_val,
        #                                        rand_state=None)
        # plt.figure()
        # if k_val is None:
        #     idx = [i for i in range(len(val_metrics))]
        # else:
        #     idx = k_val

        # for i in range(len(train_metrics)):
        #     plt.title('KNN - Validation Accuracy')
        #     # plt.plot(train_metrics[i][0], '--', label='Train {}'.format(idx[i]))
        #     plt.ylabel('Validation Accuracy')
        #     plt.xlabel('Fold')
        #     plt.plot(val_metrics[i][0], label='Val {}'.format(idx[i]))
        # plt.legend()

        # plt.figure()
        # for i in range(len(train_metrics)):
        #     plt.title('KNN - Negative Log Loss')
        #     # plt.plot(train_metrics[i][1], label='Train K={}'.format(idx[i]))
        #     plt.ylabel('Negative Log Loss')
        #     plt.xlabel('Fold')
        #     plt.plot(val_metrics[i][1], label='Val K={}'.format(idx[i]))
        # plt.legend()
 
        # mlp_val_metrics, mlp_train_metrics = hyper_mlp(h_train_data,
        #                                        h_train_labels,
        #                                        epochs=n_epochs,
        #                                        batch_size=batch_sz,
        #                                        lr=learning,
        #                                        folds=n_mlp_folds,
        #                                        rand_state=1)
        # plt.figure()
        # plt.title('MLP - Training Accuracy Per Fold')
        # for i in range(0,3):
        #     for step, acc in enumerate(mlp_train_metrics[i][0]):
        #         plt.plot(acc, label='Fold {}'.format(step+1))
        #     plt.legend()

        # plt.figure()
        # plt.title('MLP - Validation Accuracy Per Fold')
        # for i in range(0,3):
        #     for step, acc in enumerate(mlp_val_metrics[i][0]):
        #         plt.plot(acc, label='Fold {}'.format(step+1))
        #     plt.legend()

        # plt.figure()
        # plt.title('MLP - Average Training CrossEntropyLoss Per Fold')
        # for i in range(0,3):
        #     for step, loss in enumerate(mlp_train_metrics[i][1]):
        #         plt.plot(loss, label='Fold {}'.format(step+1))
        #     plt.legend()

        # plt.figure()
        # plt.title('MLP - Validation CrossEntropyLoss Per Fold')
        # for i in range(0,3):
        #     for step, loss in enumerate(mlp_val_metrics[i][1]):
        #         plt.plot(loss, label='Fold {}'.format(step+1))
        #     plt.legend()

        # plt.figure()
        # x_axis = range(1, n_epochs + 1)
        # plt.title('MLP - Average Accuracy Per Config')
        # for step, ele in enumerate(mlp_train_metrics):
        #     # print("Plotting...", ele[0])
        #     plt.plot(x_axis, ele[0], label='Config {}'.format(step+1))
        # for step, ele in enumerate(mlp_val_metrics):
        #     # print("Plotting...", ele[0])
        #     plt.plot(x_axis, ele[0], label='Config Val {}'.format(step+1))
        # plt.legend()

        # plt.figure()
        # plt.title('MLP - Average CrossEntropyLoss Per Config')
        # for step, ele in enumerate(mlp_train_metrics):
        #     # print("Plotting...", ele[1])
        #     plt.plot(x_axis, ele[1], label='Config {}'.format(step+1))
        # for step, ele in enumerate(mlp_val_metrics):
        #     # print("Plotting...", ele[1])
        #     plt.plot(x_axis, ele[1], label='Config Val {}'.format(step+1))
        # plt.legend()
            
#        val_metrics, train_metrics = kfold_knn(knn_clf,
#                                               e_train_data,
#                                               e_train_labels,
#                                               folds=n_knn_folds,
#                                               rand_state=None)
#
#        plt.figure()
#        plt.title('KNN - Accuracy')
#        plt.plot(train_metrics[0])
#        plt.plot(val_metrics[0])
#        plt.legend(['Training', 'Validation'])
#
#        plt.figure()
#        plt.title('KNN - Negative Log Loss')
#        plt.plot(train_metrics[1])
#        plt.plot(val_metrics[1])
#        plt.legend(['Training', 'Validation'])

        # MLP Cross-Validation
        mlp_val_metrics, mlp_train_metrics = kfold_mlp(neurons,
                                                        h_train_data,
                                                        h_train_labels,
                                                        epochs=n_epochs,
                                                        batch_size=batch_sz,
                                                        lr=learning,
                                                        folds=n_mlp_folds,
                                                        rand_state=1)
        print(np.asarray(mlp_val_metrics[:][0]).mean(axis=0))

        plt.figure()
        plt.title('MLP - Training Accuracy Per Fold')
        for step, acc in enumerate(mlp_train_metrics[0]):
            plt.plot(acc, label='Fold {}'.format(step+1))
        plt.legend()

        plt.figure()
        plt.title('MLP - Validation Accuracy Per Fold')
        for step, acc in enumerate(mlp_val_metrics[0]):
            plt.plot(acc, label='Fold {}'.format(step+1))
        plt.legend()

        plt.figure()
        plt.title('MLP - Average Training CrossEntropyLoss Per Fold')
        for step, loss in enumerate(mlp_train_metrics[1]):
            plt.plot(loss, label='Fold {}'.format(step+1))
        plt.legend()

        plt.figure()
        plt.title('MLP - Validation CrossEntropyLoss Per Fold')
        for step, loss in enumerate(mlp_val_metrics[1]):
            plt.plot(loss, label='Fold {}'.format(step+1))
        plt.legend()

        # Show all resulting graphs
        plt.show()
