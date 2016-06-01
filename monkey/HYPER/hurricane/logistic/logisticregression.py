#logisitic regression classifier for image classification, in scikit learn package
import os
import sys
sys.path.insert(0,"/global/u1/y/yunjie/Machine-Learning/monkey/") #add path to python
import numpy
import sklearn
import warehouse
from sklearn import linear_model as lm
from sklearn import metrics as metrics
from sklearn import cross_validation

#loading training data and test data
#positive/negative example, X: example; Y: label 
#Hurricane variable order: TMQ, V850, PSL, U850, T500, UBOT, T200, VBOT
#Atmospheric River variable order: TMQ, intersection
#Fronts variable order:Precip, sea level pressure, air temperature
Xtrain, Ytrain,  Xtest, Ytest = warehouse.load(
      ppath="/global/project/projectdirs/nervana/yunjie/climatedata",
      fname="hurricanes.h5",
      groups=['1','0'],
      npt=8000,nnt=8000,nptt=2000,nntt=2000,
      norm=0,rng_seed=2)


def hyper_opt(Cc,Sc):
    """
    optimize parameter 
    Cc: regularization
    Sc: intercept_scaling factor, not necessary, since spearmint need a vector as input, just put this parameter here
    """
    #build logistic regression model
    lgr_model=lm.LogisticRegression(C=Cc, intercept_scaling=Sc, random_state=0,solver="liblinear",
                               verbose=0,max_iter=1000) #C: inverse of strength regularization


    scores = cross_validation.cross_val_score(lgr_model,Xtrain,Ytrain,cv=3)
    score=numpy.mean(scores)
    print ("cross validation average accuracy:  %f%%" %(score*100))
    
    return (-1.0)*score  #minimize 


def main(job_id,params):
    print "Anything printed here will end up in the output directory for job #%d" %job_id
    print params
    accuracy=hyper_opt(float(params['C']),int(params['S']))

    return accuracy
