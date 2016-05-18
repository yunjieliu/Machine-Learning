#restricted boltzman machine classifier. sklearn 

import os
import sys
import numpy
import sklearn
import warehouse
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn import metrics
from sklearn.pipeline import Pipeline

#loading training data and test data
#positive/negative example, X: example; Y: label 
#Hurricane variable order: TMQ, V850, PSL, U850, T500, UBOT, T200, VBOT
#Atmospheric River variable order: TMQ, intersection
#Fronts variable order: 
Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = warehouse.load(
      ppath="/global/project/projectdirs/nervana/yunjie/climatedata",
      fname="hurricanes.h5",
      groups=['1','0'],
      npt=8000,nnt=8000,npv=1000,nnv=1000,nptt=1000,nntt=1000,
      norm=0,rng_seed=2)

#restricted boltzman machine is a feature extractor, just like convolutional neural network
#the extracted features or high level representations are passed to classifiers, such as logistic
#regression for classification. Thus this is a multi-stage training process, just like deep convolutional
#neural network


#in sklearn package, only Bernouli RBM is implemented now
#the Bernouli RBM requires the input data to be binary or within [0,1], Need to normalize data
#build the model
rbm_model=BernoulliRBM(learning_rate=0.05,n_iter=20,n_components=1000)
logistic_model=linear_model.LogisticRegression(C=10)
classifier=Pipeline(steps=[('rbm',rbm_model),('logistic',logistic_model)])

#train the model
print "---------------------------------------------"
print "training model"
classifier.fit(Xtrain,Ytrain)
params=classifier.get_params
#scores=lgr_model.score(Xtrain,Ytrain)
#print ("train set accuracy:  %0.5f%%" %(scores*100))
print "model trained prameter"
print params

#validate model
print "---------------------------------------------"
print "validating model"
Yval_pre=lgr_model.predict(Xvalid)
prob=lgr_model.predict_proba(Xvalid)
scores=lgr_model.score(Xvalid,Yvalid)

print ("validation set accuracy:  %0.5f%%" %(scores*100))

c_matrix=metrics.confusion_matrix(Yvalid,Yval_pre)
#True Negative    False Positive
#False Negative   True Positive
print ("cunfusion matrix for validation set...")
print c_matrix

