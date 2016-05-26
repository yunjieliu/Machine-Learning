#random forest classifier 
import os
import sys
import numpy
sys.path.insert(0,"/global/u1/y/yunjie/Machine-Learning/monkey/") #add path to python
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics as metrics
from sklearn import cross_validation
import warehouse

#loading training data and test data
#positive/negative example, X: example; Y: label 
#Hurricane variable order: TMQ, V850, PSL, U850, T500, UBOT, T200, VBOT
#Atmospheric River variable order: TMQ, intersection
#Fronts variable order: 
Xtrain, Ytrain,Xtest, Ytest = warehouse.load(
      ppath="/global/project/projectdirs/nervana/yunjie/climatedata",
      fname="hurricanes.h5",
      groups=['1','0'],
      npt=8000,nnt=8000,nptt=2000,nntt=2000,
      norm=0,rng_seed=2)


#build random forest classifier
def hyper_opt(ne,ms,dm,fe):
    """
    optimize parameter
    ne: number of trees in the forest   
    ms: minimal number of example required to split an internal node
    dm: max depth of tree
    fe: number of feature for finding spliting
    """
    rf_model=RandomForestClassifier(n_estimators=ne,criterion='entropy',max_features=fe, \
                                random_state=0,max_leaf_nodes=None,\
                                min_samples_split=ms,max_depth=dm,bootstrap=True,oob_score=True)

    scores = cross_validation.cross_val_score(rf_model,Xtrain,Ytrain,cv=3)
    score=numpy.mean(scores)
    print ("cross validation average accuracy:  %f%%" %(score*100))

    return score


def main(job_id,params):
    print "Anything printed here will end up in the output directory for job #%d" %job_id
    print params
    accuracy=hyper_opt(params['N'],params['M'],params['D'],params['F'])

    return accuracy
