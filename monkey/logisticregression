#logisitic regression classifier for image classification, in scikit learn package
import os
import sys
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
#Fronts variable order: 
Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = warehouse.load(
      ppath="/global/project/projectdirs/nervana/yunjie/climatedata",
      fname="hurricanes.h5",
      groups=['1','0'],
      npt=1000,nnt=1000,npv=1000,nnv=1000,nptt=1000,nntt=1000,
      norm=0,rng_seed=2)


def hyper_opt(Cc):
    #build logistic regression model
    lgr_model=lm.LogisticRegression(C=Cc, random_state=0,solver="liblinear",
                               verbose=0,max_iter=1000) #C: inverse of strength regularization


    scores = cross_validation.cross_val_score(lgr_model,Xtrain,Ytrain,cv=3)
    score=numpy.mean(scores)
    print ("cross validation average accuracy:  %f%%" %(score*100))
    
    return score 


def main(job_id,params):
    print "Anything printed here will end up in the output directory for job #%d" %job_id
    print params
    accuracy=hyper_opt(params['C'])
    return accuracy


sys.exit(0)


#fit model with data (training)
print "----------------------------------------------"
print "model fitting"
lgr_model.fit(Xtrain, Ytrain)
params=lgr_model.get_params
#scores=lgr_model.score(Xtrain,Ytrain)
#print ("train set accuracy:  %0.5f%%" %(scores*100))
print "model trained prameter"
print params

#valid
print "---------------------------------------------"
print "validating model"
Yval_pre=lgr_model.predict(Xvalid)
prob=lgr_model.predict_proba(Xvalid)

accuracy_score=metrics.accuracy_score(Yvalid,Yval_pre)
print ("validation set accuracy:  %0.5f%%" %(accuracy_score*100))
c_matrix=metrics.confusion_matrix(Yvalid,Yval_pre)
#True Negative    False Positive
#False Negative   True Positive
print ("cunfusion matrix for validation set...")
print c_matrix

sys.exit(0)
#from result of training and validating decide to do the testing or not
#test
print "---------------------------------------------"
print ("testing model ")
Ypredict=lgr_model.predict(Xtest)
prob=lgr_model.predict_proba(Xtest)
scores=lgr_model.score(Xtest,Ytest)

print("testing set accuracy:  %0.5f%%" %(scores*100)) 

matrix=metrics.confusion_matrix(Ytest,Ypredict)
#True Negative    False Positive
#False Negative   True Positive
print ("cunfusion matrix for testing set...")
print c_matrix

#compute ratio



