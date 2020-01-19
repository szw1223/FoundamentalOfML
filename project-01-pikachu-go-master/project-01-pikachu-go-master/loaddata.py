##### -! /bin/env python3

import numpy as np
import pickle
import argparse

import matplotlib.pyplot as plt

def make_parser():
    '''
        Make an argument parser to get the file paths for finding the 'data' and 'labels' in .npy file format
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Data FilePath; must be .pkl file")
    parser.add_argument("labels_path", type=str, help="Labels FilePath")
    return parser.parse_args()


def pad_data(data):
    data = [np.array(d) for d in data]
    pix = 50

    for i in range(len(data)):
        # print(i)
        x = data[i]
        # The larger axis will be up to 50 pixels, so the min_axis must be padded.
        rows = x.shape[0]
        row_pad = max((pix - rows) // 2, 0)
        row_end = max(pix - rows - row_pad // 2, 0)
        np.pad(x, ((row_pad, row_end), (0,0)), 'constant')
        
        cols = max(x.shape[1], 0)
        col_pad = max((pix - cols) // 2, 0)
        col_end = max(pix - cols - col_pad, 0)
        np.pad(x, ((0, 0), (col_pad, col_end)), 'constant')

    return data


def load_pkl(fname):
    with open(fname,'rb') as f:
        return pickle.load(f)


def save_pkl(fname,obj):
    with open(fname,'wb') as f:
        pickle.dump(obj,f)


if __name__ == '__main__':
    args = make_parser()
    data = load_pkl(args.data_path)
    labels = np.load(args.labels_path)
    lookup = {1:'a', 2:'b', 3:'c', 4:'d', 5:'h', 6:'i', 7:'j', 8:'k'}

    # Deprecated in comments
    # data = pad_data(data)
    # data = save_pkl(args.data_path, data)

    data_pair = zip(data, labels)
    # data_pair = zip(data, labels)
    with open('data_debug.txt', 'w+') as f:
        for d, c in data_pair:
            total = np.sum(np.asarray(d))
            size = d.shape[0]*d.shape[1]
            f.write("{} - {} - {}/{} ({:.4})\n".format(lookup[c], np.asarray(d).shape, total, size, total/size))

    # for i in range(0, len(easy_data), 50):
    #     d = data[i]
    #     c = labels[i]

    #     fig = plt.figure()
    #     plt.imshow(d, cmap='Greys')
    #     plt.title(lookup[c])
    #     plt.pause(1)
    #     plt.close(fig)
