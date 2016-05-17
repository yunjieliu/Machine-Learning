#support vector machine for image classification, scikit learn
import os
import sys
import sklearn
from sklearn import svm as svm
from sklearn import metrics as metrics
import warehouse

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


#build support vector machine model
#using linear kernel
print "***********************************"
print 'linear kernel'
linear_model=svm.LinearSVC()

print "-----------------------------------"
print "model fitting"
linear_model.fit(Xtrain, Ytrain)

params=linear_model.get_params
print "trained linear model parameters"
print params


print "-----------------------------------"
print "model validation"
Yval_predict=linear_model.predict(Xvalid)
scores=linear_model.score(Xvalid,Yvalid)

print ("validating set accuracy: %5f%%" %(scores*100) )

confusion_matrix=metrics.confusion_matrix(Yvalid,Yval_predict)
print "confusion matrix for validating set"
print confusion_matrix


#using non linear kernel
print "***********************************"
print 'nonlinear kernel'

nonlinear_model=svm.SVC(kernel='rbf',gamma=0.005)

print "-----------------------------------"
print "model fitting"
nonlinear_model.fit(Xtrain,Ytrain)

params=nonlinear_model.get_params
print "trained  non linear model parameters"
print params

print "-----------------------------------"
print "model validation"
Yval_predict=nonlinear_model.predict(Xvalid)
scores=nonlinear_model.score(Xvalid,Yvalid)

print ("validating set accuracy: %5f%%" %(scores*100) )

confusion_matrix=metrics.confusion_matrix(Yvalid,Yval_predict)
print "confusion matrix for validating set"
print confusion_matrix


#choose best performed model for testing




