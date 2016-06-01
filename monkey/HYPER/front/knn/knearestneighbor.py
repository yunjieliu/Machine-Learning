#K nearest neighbor classifier for image clasisfication, scikit learn
import os
import sys
import numpy
import sklearn
from sklearn import neighbors as neighbors
from sklearn import metrics as metrics
from sklearn import cross_validation
sys.path.insert(0,"/global/u1/y/yunjie/Machine-Learning/monkey/") #add path to python
import warehouse

#loading training data and test data
#positive/negative example, X: example; Y: label 
#Hurricane variable order: TMQ, V850, PSL, U850, T500, UBOT, T200, VBOT
#Atmospheric River variable order: TMQ, intersection
#Fronts variable order: 
Xtrain, Ytrain,  Xtest, Ytest = warehouse.load(
      ppath="/global/project/projectdirs/nervana/yunjie/climatedata",
      fname="fronts_all.h5",
      groups=['Front','NonFront'],
      npt=4600,nnt=4600,nptt=1000,nntt=1000,
      norm=0,rng_seed=2)

#build KNN model

def hyper_opt(nn,ls):
    """
    optimize parameter:
    nn: number of neighbors 
    ls: leaf size
    """
    K_model=neighbors.KNeighborsClassifier(n_neighbors=nn,leaf_size=ls,algorithm='auto',\
                      metric='minkowski', p=2,weights='distance')

    scores = cross_validation.cross_val_score(K_model,Xtrain,Ytrain,cv=3)
    #here we use 3 fold cross validation
    score=numpy.mean(scores)
    print ("cross validation average accuracy:  %f%%" %(score*100))

    return (-1.0)*score  #minimize

def main(job_id,params):
    print "Anything printed here will end up in the output directory for job #%d" %job_id
    print params
    accuracy=hyper_opt(int(params['nn']),int(params['ls']))
    return accuracy
