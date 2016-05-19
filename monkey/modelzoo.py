#model zoo, a collection of all interesting models for images classification from scikit learn
import time
import sklearn
from sklearn import metrics as metrics
from sklearn import cross_validation

"""
Logistic Regression
"""
def lgr(Xtrain,Ytrain):
    from sklearn import linear_model  as lm
    #build model
    lgr_model=lm.LogisticRegression(C=10, random_state=0,solver="liblinear")

    print "----------------------------------------------"
    print "model fitting"
    start=time.clock()

    #lgr_model.fit(Xtrain, Ytrain)
    #cross validation KFold
    scores = cross_validation.cross_val_score(lgr_model,Xtrain,Ytrain,cv=3)
    score=mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(scores*100))

    params=lgr_model.get_params
    print "model trained prameter"
    print params

    end=time.clock() 
    print ("training model took %f seconds" %(end-start))

    return  lgr_model


def dtr(Xtrain,Ytrain,Xvalid,Yvalid):
  
    from sklearn import tree
    #build model
    tree_model=tree.DecisionTreeClassifier()
    
    print "-----------------------------------"
    print "model fitting"

    start=time.clock()
    #tree_model.fit(Xtrain, Ytrain)

    scores = cross_validation.cross_val_score(tree_model,Xtrain,Ytrain,cv=3)
    score=mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(scores*100))


    params=tree_model.get_params
    print "trained decision tree model parameters"
    print params

    end=time.clock()
    print("training model took %f seconds" %(end-start))

    return tree_model

def knn(Xtrain,Ytrain,Xvalid,Yvalid):
    
    from sklearn import neighbors
    
    #build neighbors
    K_model=neighbors.KNeighborsClassifier(15,weights='uniform')

    print "-----------------------------------"
    print "model fitting"
    
    start=time.clock()
    #K_model.fit(Xtrain,Ytrain)

    scores = cross_validation.cross_val_score(K_model,Xtrain,Ytrain,cv=3)
    score=mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(scores*100))

    params=K_model.get_params
    print "trained KNN model parameters"
    print params

    end=time.clock()
    print("training model took %f seconds" %(end-start))
    
    return K_model

def svm_linear(Xtrain,Ytrain,Xvalid,Yvalid):

    from sklearn import svm
    
    #build models
    linear_model=svm.LinearSVC()

    print "-----------------------------------"
    print "model fitting"
    start=time.clock()
    #linear_model.fit(Xtrain, Ytrain)
    scores = cross_validation.cross_val_score(linear_model,Xtrain,Ytrain,cv=3)
    score=mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(scores*100))

    params=linear_model.get_params
    print "trained linear model parameters"
    print params

    end=time.clock()
    print("training model took %f seconds" %(end-start))

    return linear_model

def svm_nonlinear(Xtrain,Ytrain,Xvalid,Yvalid):
   
    from sklearn import svm
    
    #build models
    nonlinear_model=svm.SVC(kernel='sigmoid',gamma=0.001,C=10)

    print "-----------------------------------"
    print "model fitting"
    
    start=time.clock()
    #nonlinear_model.fit(Xtrain,Ytrain)

    scores = cross_validation.cross_val_score(nonlinear_model,Xtrain,Ytrain,cv=3)
    score=mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(scores*100))

    params=nonlinear_model.get_params
    print "trained  non linear model parameters"
    print params
 
    end=time.clock()

    print("training model took %f seconds" %(end-start)) 

    return nonlinear_model
    
def rbm(Xtrain,Ytrain,Xvalid,Yvalid):
    from sklearn import linear_model
    from sklearn.neural_network import BernoulliRBM
    from sklearn.pipeline import Pipeline

    #in sklearn package, only Bernouli RBM is implemented now
    #the Bernouli RBM requires the input data to be binary or within [0,1], Need to normalize data
    #build the model
    rbm_model=BernoulliRBM(learning_rate=0.05,n_iter=20,n_components=1000)
    logistic_model=linear_model.LogisticRegression(C=10)
    classifier=Pipeline(steps=[('rbm',rbm_model),('logistic',logistic_model)])

    print "---------------------------------------------"
    print "training model"
    
    start=time.clock()
    #classifier.fit(Xtrain,Ytrain)

    scores = cross_validation.cross_val_score(classifier,Xtrain,Ytrain,cv=3)
    score=mean(scores)
    print ("cross validation accuracy:  %0.5f%%" %(scores*100))

    params=classifier.get_params
    #scores=lgr_model.score(Xtrain,Ytrain)
    #print ("train set accuracy:  %0.5f%%" %(scores*100))
    print "model trained prameter"
    print params

    end=time.clock()
    print("training model took %s seconds" %(end-start))

    return  classifier
