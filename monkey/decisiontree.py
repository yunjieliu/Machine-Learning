#decision tree classifier for image classification. scikit learn
import os
import sys
import sklearn
from sklearn import tree as tree
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

#build the model 
tree_model=tree.DecisionTreeClassifier()

#train the model
tree_model.fit(Xtrain, Ytrain)

params=tree_model.get_params
print "trained decision tree model parameters"
print params

#validating model
print "-----------------------------------"
print "model validation"
Yval_predict=tree_model.predict(Xvalid)

scores=tree_model.score(Xvalid,Yvalid)

print ("validating set accuracy: %5f%%" %(scores*100) )

confusion_matrix=metrics.confusion_matrix(Yvalid,Yval_predict)
print "confusion matrix for validating set"
print confusion_matrix


