#master program to call and train model
import os
import sys
import numpy
import sklearn
import logging
logging.basicConfig(level=logging.INFO,filename="LOG.log",\
       filemode='w',format="%(asctime)-5s %(message)s")
import warehouse
import modelzoo

#loading training data and test data
#positive/negative example, X: example; Y: label 
#Hurricane variable order: TMQ, V850, PSL, U850, T500, UBOT, T200, VBOT
#Atmospheric River variable order: TMQ, intersection
#Fronts variable order: precip, sea level pressure, air temperature

#specify model to be used

if(len(sys.argv) < 2): #first argument will be 'self'
  print ("[Usage: Specify which model to use \n"
        + "lgr: logistic regression   \n"
        + "dtr: decision tree \n"
        + "knn: k nearest neighbor \n"
        + "svm_linear: support vector machine with linear kernel \n"
        + "svm_nonlinear: support vecor machine with non linear kernel \n"
        + "rbm: restricted Boltzman machine feature extractor +logistic regression classifier] \n"
        + "radf: random forest classifier \n" )
  sys.exit(0)

model= sys.argv[1]

#load data
Xtrain, Ytrain, Xtest, Ytest = warehouse.load(
      ppath="/global/project/projectdirs/nervana/yunjie/climatedata",
      fname="fronts_all.h5",
      groups=['Front','NonFront'],
      npt=1000,nnt=1000,nptt=1000,nntt=1000,
      norm=2,rng_seed=2)

#data information
logging.info("Training, Testing data size %d, %d" %(Xtrain.shape[0], Xtest.shape[0]))

#---------------------------------------------------------
#logistic regression

if model=="lgr":
   logging.info("UseModel: logistic regression") 
   mmodel=modelzoo.lgr(Xtrain,Ytrain)
   Ypredict=mmodel.predict(Xtest)
   scores=mmodel.score(Xtest,Ytest)
   logging.info("testing set accuracy:  %0.5f%%" %(scores*100))

#decision tree

elif model=="dtr":
  logging.info("UseModel: decision tree")
  mmodel=modelzoo.dtr(Xtrain,Ytrain)
  Ypredict=mmodel.predict(Xtest)
  scores=mmodel.score(Xtest,Ytest)
  logging.info("testing set accuracy:  %0.5f%%" %(scores*100))

#k nearest neighbor 
elif model=="knn":
  logging.info("UseModel: K nearest neighbors")
  mmodel=modelzoo.knn(Xtrain,Ytrain) 
  Ypredict=mmodel.predict(Xtest)
  scores=mmodel.score(Xtest,Ytest)
  logging.info("testing set accuracy:  %0.5f%%" %(scores*100))

#support vector machine, linear kernel
elif model=="svm_linear":
  logging.info ("UseModel: linear kernel support vector machine")
  mmodel=modelzoo.svm_linear(Xtrain,Ytrain)
  Ypredict=mmodel.predict(Xtest)
  scores=mmodel.score(Xtest,Ytest)
  logging.info("testing set accuracy:  %0.5f%%" %(scores*100))
 
#support vector machine, non linear kernel
elif model=="svm_nonlinear":
  logging.info("UseModle:  non linear kernel support vector machine")
  mmodel=modelzoo.svm_nonlinear(Xtrain,Ytrain) 
  Ypredict=mmodel.predict(Xtest)
  scores=mmodel.score(Xtest,Ytest)
  logging.info("testing set accuracy:  %0.5f%%" %(scores*100))

#restricted boltzman machine
elif model=="rbm":
  logging.info ("UseModel: restricted Boltzman machine")
  mmodel=modelzoo.rbm(Xtrain,Ytrain)
  Ypredict=mmodel.predict(Xtest)
  scores=mmodel.score(Xtest,Ytest)
  logging.info("testing set accuracy:  %0.5f%%" %(scores*100))

elif model=="radf":
  logging.info("UseModel: random forest")
  mmodel=modelzoo.radf(Xtrain,Ytrain)
  Ypredict=mmodel.predict(Xtest)
  scores=mmodel.score(Xtest,Ytest)
  logging.info("testing set accuracy:  %0.5f%%" %(scores*100))  
