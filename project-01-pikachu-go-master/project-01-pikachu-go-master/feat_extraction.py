import time
import re
import os
import argparse
import loaddata

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.morphology import closing, remove_small_objects, skeletonize, disk, square
from skimage.measure import regionprops, label


def make_parser():
    '''
        Make an argument parser to get the file paths for finding the 'data' and 'labels' in .npy file format
    '''
    parser = argparse.ArgumentParser(description='Extracts features from the test data set with the option of grabbing from the training set as well.',
                                     add_help=True)
    parser.add_argument("--train", action='store_true', help="Indicates feature extraction of training data as well as test data.")
    return parser.parse_args()


def view_data(file_path, view=False):
    '''
        Opens and (optionally) views the data in the .npy, .pkl data objects
    '''
    label_dict = {1:'a', 2:'b', 3:'c', 4:'d', 5:'h', 6:'i', 7:'j', 8:'k'}
    # Acquire all possible data.npy and label.npy files from file paths given
    data_files = []
    for dir in file_path:
        dir_listing = os.listdir(dir)
        file_pair = []

        # Find the data.npys files
        for f in dir_listing:
            match = re.match(r'(.*)data(.*)', f)
            if match != None:
                print("Got {} from directory {}".format(f, dir))
                if f.endswith('.pkl') or f.endswith('.npy'):
                    file_pair.append(dir + '/' + f)

        # Find the labels.npy files
        for f in dir_listing:
            match = re.match(r'(.*)[lL]abel(.*).npy', f)
            if match != None:
                print("Got {} from directory {}".format(f, dir))
                file_pair.append(dir + '/' + f)

        data_files.append(file_pair)
        if view:
            print(data_files)

    data_files = np.array(data_files)

    # No files found
    if np.min(data_files.shape) == 0:
        print("No '.npy' files found in the given directories; quitting data visualization")
        return

    data = []
    labels = []
    for pair in data_files:
        d = pair[0]
        if d.endswith('.pkl'):
            data.extend(loaddata.load_pkl(d))
        else:
            data.extend(np.load(d, allow_pickle=True))
        if len(pair) > 1:
            lbl = pair[1]
            labels.extend(np.load(lbl, allow_pickle=True))

    # If we want to see the binarized data and their respective labels...
    if view:
        data_pair = zip(data, labels)
        
        # Visualize all the data
        for d, lbl in data_pair:
            for i in range(len(d)):
                print("Label:", label_dict[lbl[i]])
                fig = plt.figure()
                plt.imshow(d[i], cmap="Greys")
                plt.pause(0.75)
                plt.close(fig)

    return data, labels


def pad_data(data, debug=False):
    '''
        Attempts to find nonzero pixels in a bounding box and pads all around it with white space
        to make it square
    '''
    data = [np.array(d) for d in data]
    pix = 50   # Height and Width to Achieve

    # Cut away at images based on the internal binary values - For Disproportionate Images
    for i in range(len(data)):
        d = data[i]
        fg = d > 0 

        # The following code is copied/adapted from 'hw03c.py' provided with Asgn03C
        nz_r, nz_c = fg.nonzero()
        nr, nc = fg.shape
        left, right = max(0, min(nz_c)-1), min(nc-1, max(nz_c)+1)+1
        top, bot = max(0, min(nz_r)-1), min(nr-1, max(nz_r)+1)+1

        # Grab our image window
        win = d[top:bot, left:right]
        rows = bot - top
        cols = right - left

        # Pad and center the data
        row_pad = (pix - rows) // 2
        row_end = pix - rows - row_pad
        col_pad = (pix - cols) // 2
        col_end = pix - cols - col_pad
        win = np.pad(win, ((row_pad, row_end), (col_pad, col_end)), 'constant')
        
        # Re-assign to the list
        data[i] = win

        if debug:
            print("Starter Shape: {}, {}".format(nr, nc))
            print("Nonzero Shape: {}, {}".format(rows, cols))
            print("Padding: ({}, {}), ({}, {})".format(row_pad, row_end, col_pad, col_end))
            print("New Shape: {}, {}".format(data[i].shape[0], data[i].shape[1]))
            fig = plt.figure()
            plt.imshow(data[i], cmap='Greys')
            plt.show()

    return data

def extract_sums(data, zone_factor=3):
    '''
        Feature Extraction method - Simple summation of nonzero pixels in grid
        @param data - list of data loaded from at least one data.npy file
    '''
    # Zone_factor is sqrt(number of zones), so a 5x5 would mean zone_factor = 5
    zone_size = data[0].shape[0] // zone_factor

    x = []  # List that holds training data (no labels, just features)
    for i in range(len(data)):
        feats = []   # Temporary list to hold extracted data features
        pic = data[i]
        total_nonzero = np.sum(pic.flatten())
        for r in range(zone_factor):
            row_sum = []
            row_start = zone_size * r
            for c in range(zone_factor):
                col_start = zone_size * c
                zone = pic[row_start:row_start + (zone_size + 1),
                           col_start:col_start + zone_size + 1]
                row_sum.append(np.sum(zone.flatten()))
            feats.extend(row_sum)
        # Append the features as a new data point
        if total_nonzero > 0:
            feats /= total_nonzero
        x.append(feats)

    return x

def extract_centroid(skl, props, idx=-1):
    '''
    DEPRECATED CODE: No longer used
    '''
    # centroid of first labeled object
    # centroid = props[0].centroid
    centroid = props[0].local_centroid
    q = centroid[0]
    w = centroid[1]
    c,r = find_inter(skl)
    #the first method of optimizing the centroid by obtaining the average of coordinations of all the
    #points in the character and get the ratio of the centroid and the average number
    centroidout = [0,0]
    centroidout[0] = q/r
    centroidout[1] = w/c
    ratio = centroidout[1]/centroidout[0]
    # print(ratio)
    cmin,cmax,rmin,rmax = find_inter1(skl)

    #The second method of optimizing the centroid by adding offset to the character region
    if rmax-rmin == 0 or cmax-cmin == 0 or centroidout[0] == 0:
        print("Divide by 0 in", idx)

    if rmax-rmin >= cmax-cmin:
        cmax = cmax+(rmax-rmin+cmin-cmax)/2
        cmin = cmin-(rmax-rmin+cmin-cmax)/2
    else:
        rmax = rmax+(cmax-cmin+rmin-rmax)/2
        rmin = rmin-(cmax-cmin+rmin-rmax)/2

    centroidout1 = [0,0]
    centroidout1[0] = (q-rmin)/(rmax-rmin)
    centroidout1[1] = (w-cmin)/(cmax-cmin)
    ratio1 = centroidout1[1]/centroidout1[0]
    # print(ratio1)

    return centroidout, centroidout1

def find_inter1(img):
    '''
    DEPRECATED CODE: No longer used
    '''
    cmin,cmax,rmin,rmax = 0,0,0,0
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i,j] == True:
                rmin = i
                break 
    for i in range(0,img.shape[1]):
        for j in range(0,img.shape[0]):
            if img[j,i] == True:
                cmin = i
                break    
    for i in range(img.shape[0],0,-1):
        for j in range(img.shape[1],0,-1):
            if img[i-1,j-1] == True:
                rmax = i
                break
    for i in range(img.shape[1],0,-1):
        for j in range(img.shape[0],0,-1):
            if img[j-1,i-1] == True:
                cmax = i
                break
    return cmin,cmax,rmin,rmax


def find_inter(img):
    '''
    DEPRECATED CODE: No longer used
    '''
    r,c,n = 0,0,0
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i,j] == True:
                r += i
                c += j
                n += 1
    r = r/n
    c = c/n
    return r,c

def feature_ext(data, debug=False):
    '''
        Extract features from the provided data, which is expected to be a list of numpy arrays
    '''

    # Focus on small sub-images of the data to close gaps within those areas.
    tile_size = 10
    neighbor = 5
    win_size = 6
    min_comp_size = 10
    n_hood = square(neighbor)
    # Make neighborhood into diamond shape; only works for odd sizes
    for i in range(neighbor // 2):
        n_hood[i,:neighbor // 2-i] = n_hood[i, neighbor//2 + 1+i:] = 0
        n_hood[-1-i,:neighbor // 2-i] = n_hood[-1-i, neighbor//2 + 1+i:] = 0
        n_hood[:neighbor//2-i, 0] = n_hood[neighbor//2 +1+i:, 0] = 0
        n_hood[:neighbor//2-i, -1-i] = n_hood[neighbor//2 +1+i:, -1-i] = 0
    
    # Extract features for every single sample
    feats = []
    for i in tqdm(range(len(data))):
        image = data[i]

        # Close gaps and skeletonize to make characters stroke thickness invariant
        for r in range (0,image.shape[0],win_size):
            for c in range(0,image.shape[1],win_size):
                row_lim = min(r+win_size, image.shape[0])
                c_lim = min(c+win_size, image.shape[1])
                closing(image[r:row_lim,c:c_lim],n_hood,out=image[r:row_lim,c:c_lim])
        closed = image
        remove_small_objects(closed, min_comp_size, in_place=True)
        skelly = skeletonize(closed)

        label_img = label(skelly, connectivity=skelly.ndim)
        props = regionprops(label_img)

        if len(props) < 1:
            # TODO - Does this change make sense? Or should it instead be that we throw out this index?
            temp_feats = [0 for i in range(13)]
        else:
            # Extract region properties - Centroid, moments, inertia, eccentricity
            center1 = props[0].local_centroid
            eccentricity = props[0].eccentricity
            moments_hu = props[0].moments_hu
            inertia = props[0].inertia_tensor.flatten()
            inertia_data = np.array([inertia[0], inertia[1], inertia[2]])   # Remove one of the I_xy

            temp_feats = [center1[0], center1[1], eccentricity]
            temp_feats.extend(moments_hu)
            temp_feats.extend(inertia_data)

        feats.append(temp_feats)

        # DEBUGGING: Plot the morphology changes as one plot, including skelly
        if debug:
            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(image, cmap='gray', aspect='equal',
                            vmin=0, vmax=1)
            ax[0].set_title('Original', fontsize=16)
            ax[0].axis('off')

            ax[1].imshow(closed, cmap='gray', aspect='equal',
            vmin=0, vmax=1)
            ax[1].set_title('Closing', fontsize=16)
            ax[1].axis('off')

            ax[2].imshow(skelly, cmap='gray', aspect='equal',
                            vmin=0, vmax=1)
            ax[2].set_title('skeleton', fontsize=16)
            ax[2].axis('off')
            fig.suptitle("Text Detection", fontsize=18)
            fig.tight_layout(rect=(0, 0, 1, 0.88))
            plt.show()

    return feats

def stats_feat_ext(feats, labels):
    '''
        Looks at statistics of the data with features extracted and prints them to console.
        ie. Average, median, mode, max, min

        (Unfortunately was never used)
    '''
    label_stats = []
    # Make sure we can use the iterables as np arrays
    feats = np.array(feats)
    labels = np.array(labels)

    # Per each class, get the statistics
    for c in set(labels):
        statistics = []
        # Mask out the data by label and get stats
        c_idx = np.where(labels == c)
        c_list = feats[c_idx]
        statistics.append(np.average(c_list, axis=0))
        statistics.append(np.median(c_list, axis=0))
        statistics.append(stats.mode(c_list, axis=0)[0])
        statistics.append(np.max(c_list, axis=0))
        statistics.append(np.min(c_list, axis=0))

        # Add the specific statistics to the list
        label_stats.append(statistics)
    
    for c in len(set(labels)):
        print(c, end="\n\n")


if __name__ == '__main__':
    args = make_parser()
    lookup = {1:'a', 2:'b', 3:'c', 4:'d', 5:'h', 6:'i', 7:'j', 8:'k'}

    test_data = ['test_data']
    test_dir = 'test_data/test_feat'
    test_data, test_labels = view_data(test_data)

    # Extract Features
    print("Grabbing test data set features...")
    test_data = pad_data(test_data)
    test_feats = np.array(feature_ext(test_data, debug=False))
    test_sums = np.array(extract_sums(test_data))
    test_feats = np.hstack((test_feats, test_sums))

    # Save extracted features into .npy files for future loading
    print("Size of test data set:", test_feats.shape)
    np.save(test_dir + '/' + 'test_data.npy', test_feats)
    np.save(test_dir + '/' + 'test_labels.npy', test_labels)

    # Repeat the same steps if we decide that we'd also like to extract features for training
    if args.train:
        easy_data = ['train_data/easy_set']
        hard_data = ['train_data/hard_set']
    
        easy_dir = 'train_data/easy_set/easy_feat'
        hard_dir = 'train_data/hard_set/hard_feat'
    
        # Visualize all available data - DON'T SHUFFLE DATA W/IN FEAT_EXT METHODS
        e_data, e_labels = view_data(easy_data)
        h_data, h_labels = view_data(hard_data)

        print("Grabbing easy data set features...")
        e_data = pad_data(e_data)
        e_feats = np.array(feature_ext(e_data, debug=False))
        e_feat_sums = np.array(extract_sums(e_data))
        e_feats = np.hstack((e_feats, e_feat_sums))

        print("Grabbing hard data set features...")
        h_data = pad_data(h_data)
        h_feats = np.array(feature_ext(h_data, debug=False))
        h_feat_sums = np.array(extract_sums(h_data))
        h_feats = np.hstack((h_feats, h_feat_sums))

        # Compile all features into a single feature vector per each sample
        print("Size of easy data set:", e_feats.shape)
        print("Size of hard data set:", h_feats.shape)

        # print("Send all features to a document...")
        e_feats_pair = zip(e_feats, e_labels)
        with open('easy_feat_debug.txt', 'w+') as f:
            for feat, c in e_feats_pair:
                f.write("{} || ".format(lookup[c]))
                for ele in feat:
                    f.write("{:.4} | ".format(ele))
                f.write("\n")

        h_feats_pair = zip(h_feats, h_labels)
        with open('hard_feat_debug.txt', 'w+') as f:
            for feat, c in h_feats_pair:
                f.write("{} || ".format(lookup[c]))
                for ele in feat:
                    f.write("{:.4} | ".format(ele))
                f.write("\n")

        np.save(easy_dir + '/' + 'train_easy_data.npy', e_feats)
        np.save(easy_dir + '/' + 'train_easy_labels.npy', e_labels)

        print(h_feats[0:3])
        np.save(hard_dir + '/' + 'train_hard_data.npy', h_feats)
        np.save(hard_dir + '/' + 'train_hard_labels.npy', h_labels)
