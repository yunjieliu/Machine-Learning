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
      ppath="/global/project/projectdirs/nervana/yunjie/climatedata/new_merge/",
      fname="atmospheric_river_us+eu+landsea_sep10.h5",
      groups=['AR','Non_AR'],
      npt=5800,nnt=5500,nptt=1000,nntt=1000,
      norm=0,rng_seed=2)



#build support vector machine model

def hyper_opt(Cc,ga):
    """
    optimize parameters
    Cc: regularization parameter
    ga: gamma parameter
    """
    svm_model=svm.SVC(C=Cc,kernel='rbf',degree=4,gamma=ga,coef0=1, cache_size=200,max_iter=1000,random_state=0,tol=0.0001)

    scores = cross_validation.cross_val_score(svm_model,Xtrain,Ytrain,cv=3)
    score=numpy.mean(scores)
    print ("cross validation average accuracy:  %f%%" %(score*100))

    return score


def main(job_id,params):
    print "Anything printed here will end up in the output directory for job #%d" %job_id
    print params
    accuracy=hyper_opt(params['C'],params['G'])

    return accuracy

