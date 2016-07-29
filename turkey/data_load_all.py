"""
Module:
load training data and split into training, validating and testing
include: hurricane, atmospheric river and fronts
"""

import os
import numpy
import h5py
import logging
import sklearn
from sklearn import preprocessing
#import ipdb

logger=logging.getLogger('load_data')
handler=logging.StreamHandler()
formatter=logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel("INFO")

#define function to load the data 

def happy_loader(rng_seed,fname,groups,train_num_p,valid_num_p,test_num_p,n_class,now_class,n_type,normalize):
    """
    load hurricane, atmospheric river, fronts data and split into train, valid and test set
    """
    numpy.random.seed(rng_seed)
    #NOTE, NEON seems has problmes of the RNG_SEED in the backend, I am explicitly specify the seeds here
    #HERE we want the labels be the same format as CIFAR10
    """
    0 1 0 0 0
    0 0 0 1 0
    0 1 0 0 0
    1 0 0 0 0
    ...
    ...
    """ 

    logger.info("training  data...%s" %fname)
    data=h5py.File(fname,"r")
    hurricane_positive=data[groups[0]]

    #generally follow the 80%-20% rule, X is data, Y is label
    X_train=hurricane_positive[:train_num_p]
    #create label for classes, standard CIFAR style
    Y_train=hotcode(train_num_p,n_class,now_class)

    X_valid=hurricane_positive[train_num_p:train_num_p+valid_num_p]
    Y_valid=hotcode(valid_num_p,n_class,now_class)

    X_test=hurricane_positive[train_num_p+valid_num_p:train_num_p+valid_num_p+test_num_p]
    Y_test=hotcode(test_num_p,n_class,now_class)

    #normalize data if normalize is needed
    if normalize:
       if abs(n_type -1)==0: #global contrast
          X_train=global_contrast_norm(X_train)
          X_valid=global_contrast_norm(X_valid)
          X_test=global_contrast_norm(X_test)
       elif abs(n_type -2)==0: #standard norm 
          X_train=stand_norm(X_train)
          X_valid=stand_norm(X_valid)
          X_test=stand_norm(X_test)
       elif abs(n_type -3)==0: #sklearn style l1/l2 norm
          X_train=norm_norm(X_train)
          X_valid=norm_norm(X_valid)
          X_test=norm_norm(X_test)

    # randomly shuffle data, mixing positive and negative example
    X_train,Y_train=rand_data(X_train,Y_train)
    X_valid,Y_valid=rand_data(X_valid,Y_valid)
    X_test,Y_test=rand_data(X_test,Y_test)

    #flat all input images into feature vector (The ner version of NEON requires data be presented this way)
    #X_train=fllat(X_train)
    #X_valid=fllat(X_valid)
    #X_test=fllat(X_test)
        
    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)

def fllat(A):
    """
    flat input images into a one dimensional feature vector
    pay attention to the nD array structure
    A: image data
    """
    shp=A.shape
    B=A.reshape(shp[0],-1) #flat all except the first dimension
    return B

def rand_data(A,B):
    """
    randomly mixing/shuffing the positive and negative example
    A: image data
    B: corresponding label
    """
    logger.info("randomly shuffle data...")
    ss=range(len(A))
    numpy.random.shuffle(ss)
    A=A[ss]
    B=B[ss]    

    return A, B


"""
I previously found that classification model performance and modle overfitting 
is sensitive to the data normalizaion. Here try several normalization technique
1) simple feature scaling (scale to [0,1] or [-1,1])
2) gobal contrast normalization
3) l1 or l2 norm normalization
"""

def global_contrast_norm(A, scale=1.0,min_divisor=1e-8):
    """
    ###this function is the same as in NEON source code
    Subtract mean and normalize by vector norm [normalize accross channel]
    A: image data
    """
    
    logger.info("do global contrast normalization...")
    A = A - A.mean(axis=1)[:, numpy.newaxis]

    normalizers = numpy.sqrt((A ** 2).sum(axis=1)) / scale
    normalizers[normalizers < min_divisor] = 1.

    A /= normalizers[:, numpy.newaxis]

    return A

def stand_norm(A):
    """
    subtract mean and divide by standard deviation of each channel
    A: image data
    """
    logger.info("do standard normalization...")

    sh=A.shape
    A =A.reshape(sh[0],sh[1],-1) #flat feature of each channel
    A =A -A.mean(axis=2)[:,:,numpy.newaxis] #numpy.newaxis makes a new matrix dimension
    stdd=A.std(axis=2)[:,:,numpy.newaxis]
    stdd[stdd<1e-8]=1.
    A /=stdd
    #A /= A.std(axis=2)[:,:,numpy.newaxis]

    A=A.reshape(sh)
       
    return A

def norm_norm(A):
    """
    l1 norm of input data (scikit learn)
    A: image data
    """
    logger.info("do l1/l2 norm normalization...")
    sh=A.shape
    A=A.reshape(sh[0],sh[1],-1)
    for i in range(sh[0]):
        A[i]=preprocessing.normalize(A[i], axis=1,norm="l2")
         
    A=A.reshape(sh)
    return A

def hotcode(a,n,nn):
    """
    a: number of positive examples, encode 1
    n: number of classes
    nn: which class to look at
    return code of class labeling (CIFAR like)
    """
    AA=numpy.zeros((a,n))
    AA[:,nn]=1.0
    return AA




