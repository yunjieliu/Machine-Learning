#Data ware house
#loading data, random shuffle and normalization
import os
import sys
import h5py
import numpy
import sklearn
from sklearn import preprocessing

def load(ppath,fname,groups,npt,nnt,npv,nnv,nptt,nntt,norm=0,rng_seed=1):  
    """
    loading data from specified source
    rng_seed: random number generator seed, rng_seed=1 by default
    ppath: path to the data file
    fname: data file name
    npt,npv,nptt: number of positive training,validating and testing example
    nnt,nnv,nntt: number of negative training,validating and testing example
    norm: how to normalize data, default norm=0, no normalization
    """
    #read in data file 
    ff=os.path.join(ppath,fname)
    print "ready for data file..." +ff
    data=h5py.File(ff,'r')

    positive=data[groups[0]]
    negative=data[groups[1]]
    
    
    #extract train, valid and test data set
    #X: example, Y: label
    Xtrain=numpy.vstack((positive[:npt],negative[:nnt]))
    Ytrain=numpy.hstack((numpy.ones(npt),numpy.zeros(nnt)))
    
    
    Xvalid=numpy.vstack((positive[npt:npt+npv],negative[nnt:nnt+nnv]))
    Yvalid=numpy.hstack((numpy.ones(npv),numpy.ones(nnv)))

    Xtest=numpy.vstack((positive[npt+npv:npt+npv+nptt],negative[nnt+nnv:nnt+nnv+nntt]))
    Ytest=numpy.hstack((numpy.ones(nptt),numpy.zeros(nntt)))

    
    Xtrain=Xtrain[:,2,...]
    Xvalid=Xvalid[:,2,...]
    Xtest=Xtest[:,2,...]
    #normalize train, valid and test set
    if norm==0:
       pass
    elif norm==1:
       Xtrain=stand_norm(Xtrain)
       Xvalid=stand_norm(Xvalid)
       Xtest=stand_norm(Xtest)

    #randomly shuffle and mixing data
    #Xtrain,Ytrain=rand_shuffle(Xtrain,Ytrain) 
    #Xvalid,Yvalid=rand_shuffle(Xvalid,Yvalid)
    #Xtest,Ytest=rand_shuffle(Xtest,Ytest)

    #flatten into 1D array 
    
    Xtrain=fllat(Xtrain)
    Xvalid=fllat(Xvalid)
    Xtest=fllat(Xtest)

    return Xtrain,Ytrain, Xvalid, Yvalid, Xtest,Ytest


#other useful functions 
def fllat(A):
    """
    flat input images into a one dimensional feature vector
    pay attention to the nD array structure
    A: image data
    """
    shp=A.shape
    B=A.reshape(shp[0],-1) #flat all except the first dimension
    return B

def rand_shuffle(A,B):
    """
    randomly mixing/shuffing the positive and negative example
    A: image data
    B: corresponding label
    """
    print("randomly shuffle data...")
    ss=range(len(A))
    numpy.random.shuffle(ss)
    A=A[ss]
    B=B[ss]

    return A, B

def stand_norm(A):
    """
    subtract mean and divide by standard deviation of each channel
    A: image data
    """
    print("do standard normalization...")

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
