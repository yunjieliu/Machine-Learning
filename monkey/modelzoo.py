#model zoo, a collection of all interesting models for images classification from scikit learn
import time
import sklearn
from sklearn import metrics as metrics
from sklearn import cross_validation
import numpy

"""
Logistic Regression
"""
def lgr(Xtrain,Ytrain):

    #class sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    from sklearn import linear_model  as lm
    #build model
    lgr_model=lm.LogisticRegression(C=5, penalty='l2',max_iter=1000,random_state=0,solver="liblinear")

    print "----------------------------------------------"
    print "model fitting"
    start=time.clock()

    #lgr_model.fit(Xtrain, Ytrain)
    #cross validation KFold
    scores = cross_validation.cross_val_score(lgr_model,Xtrain,Ytrain,cv=3)
    print ("3 fold cross validation accuracy ")
    print scores
    score=numpy.mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(score*100))

    lgr_model.fit(Xtrain,Ytrain)
    params=lgr_model.get_params
    print "model trained prameter"
    print params

    end=time.clock() 
    print ("training model took %f seconds" %(end-start))

    return  lgr_model


def dtr(Xtrain,Ytrain):

    #class sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None, presort=False)
  
    from sklearn import tree
    #build model
    tree_model=tree.DecisionTreeClassifier(criterion='entropy',max_features='sqrt',max_depth=None,min_samples_split=10,min_samples_leaf=24,min_weight_fraction_leaf=0.2, \
                                           max_leaf_nodes=240,random_state=0)
    
    print "-----------------------------------"
    print "model fitting"

    start=time.clock()
    #tree_model.fit(Xtrain, Ytrain)

    scores = cross_validation.cross_val_score(tree_model,Xtrain,Ytrain,cv=3)
    print ("3 fold cross validation accuracy ")
    print scores
    score=numpy.mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(score*100))
    
    tree_model.fit(Xtrain,Ytrain)
    params=tree_model.get_params
    print "trained decision tree model parameters"
    print params

    end=time.clock()
    print("training model took %f seconds" %(end-start))

    return tree_model

def knn(Xtrain,Ytrain):
    #class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)

    from sklearn import neighbors
    
    #build neighbors
    K_model=neighbors.KNeighborsClassifier(n_neighbors=16,weights='distance',algorithm='auto',leaf_size=16,p=2,metric='minkowski')

    print "-----------------------------------"
    print "model fitting"
    
    start=time.clock()
    #K_model.fit(Xtrain,Ytrain)

    scores = cross_validation.cross_val_score(K_model,Xtrain,Ytrain,cv=3)
    print ("3 fold cross validation accuracy ")
    print scores
    score=numpy.mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(score*100))
    
    K_model.fit(Xtrain,Ytrain)
    params=K_model.get_params
    print "trained KNN model parameters"
    print params

    end=time.clock()
    print("training model took %f seconds" %(end-start))
    
    return K_model

def svm_linear(Xtrain,Ytrain):
    #class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

    from sklearn import svm
    
    #build models
    linear_model=svm.LinearSVC(penalty='l2',loss='hinge',multi_class='ovr',C=5.0,random_state=0,max_iter=1000)

    print "-----------------------------------"
    print "model fitting"
    start=time.clock()
    #linear_model.fit(Xtrain, Ytrain)
    scores = cross_validation.cross_val_score(linear_model,Xtrain,Ytrain,cv=3)
    print ("3 fold cross validation accuracy ")
    print scores
    score=numpy.mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(score*100))

    linear_model.fit(Xtrain,Ytrain)
    params=linear_model.get_params
    print "trained linear model parameters"
    print params

    end=time.clock()
    print("training model took %f seconds" %(end-start))

    return linear_model

def svm_nonlinear(Xtrain,Ytrain):
    #class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
       
    from sklearn import svm
    
    #build models
    nonlinear_model=svm.SVC(kernel='poly',degree=4,gamma=0.001,coef0=1, cache_size=200,max_iter=1000,random_state=0)

    print "-----------------------------------"
    print "model fitting"
    
    start=time.clock()
    #nonlinear_model.fit(Xtrain,Ytrain)

    scores = cross_validation.cross_val_score(nonlinear_model,Xtrain,Ytrain,cv=3)
    print ("3 fold cross validation accuracy ")
    print scores
    score=numpy.mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(score*100))

    nonlinear_model.fit(Xtrain,Ytrain)
    params=nonlinear_model.get_params
    print "trained  non linear model parameters"
    print params
 
    end=time.clock()

    print("training model took %f seconds" %(end-start)) 

    return nonlinear_model
    
def rbm(Xtrain,Ytrain):
    #class sklearn.neural_network.BernoulliRBM(n_components=256, learning_rate=0.1, batch_size=10, n_iter=10, verbose=0, random_state=None)

    from sklearn import linear_model
    from sklearn.neural_network import BernoulliRBM
    from sklearn.pipeline import Pipeline

    #in sklearn package, only Bernouli RBM is implemented now
    #the Bernouli RBM requires the input data to be binary or within [0,1], Need to normalize data
    #build the model
    rbm_model=BernoulliRBM(n_components=64,learning_rate=0.08,n_iter=50,batch_size=100,random_state=0)
    logistic_model=linear_model.LogisticRegression(C=10)
    classifier=Pipeline(steps=[('rbm',rbm_model),('logistic',logistic_model)])

    print "---------------------------------------------"
    print "training model"
    
    start=time.clock()
    #classifier.fit(Xtrain,Ytrain)

    scores = cross_validation.cross_val_score(classifier,Xtrain,Ytrain,cv=3)
    print ("3 fold cross validation accuracy ")
    print scores
    score=numpy.mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(score*100))

    classifier.fit(Xtrain,Ytrain)
    params=classifier.get_params
    #scores=lgr_model.score(Xtrain,Ytrain)
    #print ("train set accuracy:  %0.5f%%" %(scores*100))
    print "model trained prameter"
    print params

    end=time.clock()
    print("training model took %s seconds" %(end-start))

    return  classifier

def radf(Xtrain, Ytrain):
    #class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None
    from sklearn.ensemble import RandomForestClassifier
    rf_model=RandomForestClassifier(n_estimators=16,criterion='entropy',max_features='sqrt',bootstrap=True,oob_score=True)

    print "---------------------------------------------"
    print "training model"

    start=time.clock()
    #classifier.fit(Xtrain,Ytrain)

    scores = cross_validation.cross_val_score(rf_model,Xtrain,Ytrain,cv=3)
    print ("3 fold cross validation accuracy ")
    print scores
    score=numpy.mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(score*100))

    rf_model.fit(Xtrain,Ytrain)
    params=rf_model.get_params
    #scores=lgr_model.score(Xtrain,Ytrain)
    #print ("train set accuracy:  %0.5f%%" %(scores*100))
    print "model trained prameter"
    print params

    end=time.clock()
    print("training model took %s seconds" %(end-start))

    return  rf_model

