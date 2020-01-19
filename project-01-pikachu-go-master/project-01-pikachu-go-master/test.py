import argparse
import numpy as np
import KaiyangClassifier
import torch

import loaddata
import feat_extraction

def make_parser():
    '''
        Make an argument parser to get the file paths for finding the 'data' and 'labels' in .npy file format
    '''
    parser = argparse.ArgumentParser(description="Test classifiers on easy and hard data sets.",
                                     add_help=True)
    # parser.add_argument('--labels', action='store_true', help="Checks the predicted values against true labels. (Deprecated)")
    parser.add_argument('--hard', action='store_true', help="Classify the hard data set (instead of the easy set).")
    parser.add_argument("clf", type=str, nargs='+', help="Classifier used. Needs to be .pkl file.")
    parser.add_argument("data_file", type=str, nargs='+', help="Absolute or relative path to test data file. MUST be a .pkl file.")
    return parser.parse_args()


def norm_data(feat_x):
    '''
        Normalize all features to a Gaussian(0, 1) random variable
    '''
    feat_x = np.array(feat_x)

    # Data must be arranged as matrix of size (# samples) x (# features)
    mu = feat_x.mean(axis=0)
    sigma = feat_x.std(axis=0)

    # Possibly values at variance of 0?
    bad_sigma = np.where(sigma == 0)
    sigma[bad_sigma] = 1

    feat_x = (feat_x - mu) / sigma
    return feat_x


def test_easy_clf(fname, data, labels):
    clf = loaddata.load_pkl(fname)
    data = norm_data(data)
    pred_y = clf.predict(data)

    pred_acc = -1.0
    if labels is not None:
        pred_acc = sum(pred_y == labels) / len(labels)

    return pred_y, pred_acc


def test_hard_clf(fname, data, labels):
    clf = loaddata.load_pkl(fname)
    data = norm_data(data)

    data = torch.tensor(data).float()
    pred_y = clf.predict(data)

    # Difference between values to be considered close enough to not be sure of class
    eps = 0.00005
    pred_labels = []
    for ele in pred_y:
        tensor_max = ele.max(0)
        max_val = tensor_max[0]
        temp_label = tensor_max[1]
        for i in range(len(ele)):
            if temp_label != i and abs(max_val - ele[i]) <= eps:
                temp_label = torch.tensor(-2)
        pred_labels.append(temp_label.item() + 1)
    
    # If labels provided, get the accuracy of the model
    pred_acc = -1.0
    if labels is not None:
        pred_acc = sum(pred_labels == labels) / len(labels)

    return pred_labels, pred_acc


if __name__ == '__main__':
    args = make_parser()

    # Load the test data set
    test_data = loaddata.load_pkl(args.data_file[0])

    # Extract Features
    print("Grabbing test data set features...")
    test_data = feat_extraction.pad_data(test_data)
    test_feat = np.array(feat_extraction.feature_ext(test_data, debug=False))
    test_sums = np.array(feat_extraction.extract_sums(test_data))
    test_feat = np.hstack((test_feat, test_sums))
    print("Done with feature extraction.")

    # Save extracted features into .npy files for future loading
    print("Size of test data set:", test_feat.shape)

    test_labels = None
    # if args.labels:
    #     test_labels = np.load('test_data/test_feat/test_labels.npy')
    
    # Check if they had the hard data set or not
    print("Predicting classes...")
    if args.hard:
        pred_y, pred_acc = test_hard_clf(args.clf[0], test_feat, test_labels)
    else:
        pred_y, pred_acc = test_easy_clf(args.clf[0], test_feat, test_labels)