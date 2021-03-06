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
Xtrain, Ytrain,  Xtest, Ytest = warehouse.load(
      ppath="/global/project/projectdirs/nervana/yunjie/climatedata/new_merge/",
      fname="atmospheric_river_us+eu+landsea_sep10.h5",
      groups=['AR','Non_AR'],
      npt=5800,nnt=5500,nptt=1000,nntt=1000,
      norm=2,rng_seed=2)



#build random forest classifier
def hyper_opt(ne,ms):
    """
    optimize parameter
    ne: number of trees in the forest   
    ms: minimal number of example required to split an internal node
    """
    rf_model=RandomForestClassifier(n_estimators=ne,criterion='entropy',max_features='sqrt', \
                                random_state=0,max_leaf_nodes=None,\
                                min_samples_split=ms,max_depth=None,bootstrap=True,oob_score=True)

    scores = cross_validation.cross_val_score(rf_model,Xtrain,Ytrain,cv=3)
    score=numpy.mean(scores)
    print ("cross validation average accuracy:  %f%%" %(score*100))

    return (-1.0)*score #minimize


def main(job_id,params):
    print "Anything printed here will end up in the output directory for job #%d" %job_id
    print params
    accuracy=hyper_opt(int(params['N']),int(params['M']))

    return accuracy
