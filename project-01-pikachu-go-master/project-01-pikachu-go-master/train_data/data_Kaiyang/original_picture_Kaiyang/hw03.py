# -*- coding: utf-8 -*-
"""
File:   hw03C.py
"""

"""
====================================================
================ Import Packages ===================
====================================================
"""
import sys

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import skimage.filters as filt

"""
====================================================
================ Define Functions ==================
====================================================
"""

def process_image(in_fname,out_fname,debug=False):

    # load image
    x_in = np.array(Image.open(in_fname))

    # convert to grayscale
    x_gray = 1.0-rgb2gray(x_in)

    if debug:
        plt.figure(1)
        plt.imshow(x_gray)
        plt.title('original grayscale image')
        plt.show()

    # threshold to convert to binary
    thresh = filt.threshold_minimum(x_gray)
    fg = x_gray > thresh

    if debug:
        plt.figure(2)
        plt.imshow(fg)
        plt.title('binarized image')
        plt.show()

    # find bounds
    nz_r,nz_c = fg.nonzero()
    n_r,n_c = fg.shape
    l,r = max(0,min(nz_c)-1),min(n_c-1,max(nz_c)+1)+1
    t,b = max(0,min(nz_r)-1),min(n_r-1,max(nz_r)+1)+1

    # extract window
    win = fg[t:b,l:r]

    if debug:
        plt.figure(3)
        plt.imshow(win)
        plt.title('windowed image')
        plt.show()

    # resize so largest dim is 48 pixels 
    max_dim = max(win.shape)
    new_r = int(round(win.shape[0]/max_dim*48))
    new_c = int(round(win.shape[1]/max_dim*48))

    win_img = Image.fromarray(win.astype(np.uint8)*255)
    resize_img = win_img.resize((new_c,new_r))
    resize_win = np.array(resize_img).astype(bool)

    # embed into output array with 1 pixel border
    out_win = np.zeros((resize_win.shape[0]+2,resize_win.shape[1]+2),dtype=bool)
    out_win[1:-1,1:-1] = resize_win

    if debug:
        plt.figure(4)
        plt.imshow(out_win,cmap='Greys')
        plt.title('resized windowed image')
        plt.show()

    #save out result as numpy array
    np.save(out_fname,out_win)

"""
====================================================
========= Generate Features and Labels =============
====================================================
"""

if __name__ == '__main__':

    # To not call from command line, comment the following code block and use example below 
    # to use command line, call: python hw03.py K.jpg output

    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print('usage: {} <in_filename> <out_filename> (--debug)'.format(sys.argv[0]))
        sys.exit(0)
    
    in_fname = sys.argv[1]
    out_fname = sys.argv[2]

    if len(sys.argv) == 4:
        debug = sys.argv[3] == '--debug'
    else:
        debug = False


#    #e.g. use
#    process_image('C:/Desktop/K.jpg','C:/Desktop/output.npy',debug=True)
    process_image(in_fname,out_fname)
    
    
filename=['a','b','c','d','h','i','j','k']
data=[]
labels=[]
for i in range(0,8):
    for j in range(0,10):
        data_temp=[]
        inputname=filename[i]+'('+str(j)+')'+'.jpeg'
        outputname=filename[i]+'_'+str(j)
        process_image(inputname,outputname)
        data_temp=np.load(outputname+'.npy')
        data.append(data_temp)
        if 'a' in(filename[i]):
            labels.append('1')
        elif 'b' in(filename[i]):
            labels.append('2')
        elif 'c' in(filename[i]):
            labels.append('3')
        elif 'd' in(filename[i]):
            labels.append('4')
        elif 'h' in(filename[i]):
            labels.append('5')
        elif 'i' in(filename[i]):
            labels.append('6')
        elif 'j' in(filename[i]):
            labels.append('7')
        elif 'k' in(filename[i]):
            labels.append('8')
np.save('data.npy',data)
np.save('labels.npy',labels)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:44:53 2019
Name: PCA
@author: kaiyanghan
"""

import numpy as np
X=np.array([[2,3,3,4,5,7],[2,4,5,5,6,8]])
z2=X@X.T
z1=X.T@X
eig_val1,eig_vec1=np.linalg.eig(z1)
eig_val2,eig_vec2=np.linalg.eig(z2)
eig_pairs2 = [(np.abs(eig_val2[i]), eig_vec2[:,i]) for i in range(len(eig_val2))]
eig_pairs2.sort(reverse=True)
feature2=eig_pairs2[0][1]
new_data=np.dot(feature2,X)
x=X[0,:]
y=X[1,:]
mean_x=np.mean(x)
mean_y=np.mean(y)
scaled_x=x-mean_x
scaled_y=y-mean_y
scaled_x=scaled_x/np.sqrt(np.var(scaled_x))
scaled_y=scaled_y/np.sqrt(np.var(scaled_y))
data=np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])
#
#cov=data.T@data
cov=np.cov(data.T)
eig_val, eig_vec = np.linalg.eig(cov)
#
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)
feature=eig_pairs[0][1]
new_data_reduced=np.dot(feature,data.T)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:11:22 2019
Name: EM
    
@author: kaiyanghan
"""
import numpy as np
y=[1,1,0,1,0,0,1,0,1,1]
pai=[0.5]
p=[0.5]
q=[0.5]
miu=[]
i=0
for i in range(0,20):
    miutemp=[]
    minusmiu=[]
    for j in range(0,10):
        a=pai[i]*pow(p[i],y[j])*pow((1-p[i]),(1-y[j]))
        b=(1-pai[i])*pow(q[i],y[j])*pow((1-q[i]),(1-y[j]))
        miutemp.append(a/(a+b))
        minusmiu.append(1-(a/(a+b)))
    pai.append(np.sum(miutemp)/10)
    p.append(np.sum(np.array(miutemp)*np.array(y))/np.sum(np.array(miutemp)))
    q.append(np.sum(np.array(minusmiu)*np.array(y))/np.sum(np.array(minusmiu)))
    miu.append(miutemp)