#support vector machine for image classification, scikit learn
import os
import sys
import numpy
sys.path.insert(0,"/global/u1/y/yunjie/Machine-Learning/monkey/") #add path to python
import sklearn
from sklearn import svm as svm
from sklearn import metrics as metrics
from sklearn import cross_validation
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


#build support vector machine model

def hyper_opt(Cc,Sc):
    """
    optimize parameters
    Cc: regularization parameter
    Sc: intercept scaling 
    """
    linear_model=svm.LinearSVC(penalty='l2',loss='hinge',multi_class='ovr',C=Cc, \
                 fit_intercept=True, intercept_scaling=Sc,random_state=0,max_iter=1000,tol=0.0001)


    scores = cross_validation.cross_val_score(linear_model,Xtrain,Ytrain,cv=3)
    score=numpy.mean(scores)
    print ("cross validation average accuracy:  %f%%" %(score*100))

    return (-1.0)*score #minimize


def main(job_id,params):
    print "Anything printed here will end up in the output directory for job #%d" %job_id
    print params
    accuracy=hyper_opt(float(params['C']),int(params['S']))

    return accuracy

