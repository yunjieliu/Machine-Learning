"""
Theano Logistic for CIFAR 10 dataset
"""
import os,sys
import numpy
import theano
from theano import tensor
import load

#training, testing data
X_train, Y_train, X_test, Y_test= load.cifar10(dtype=theano.config.floatX)

print "training data ready!"
#the training data are flattend into 1D array
label_train=numpy.argmax(Y_train,axis=1)

#define symbolic in theano
x=tensor.matrix()
t=tensor.matrix()


#define the logistic regression model
#if it is binary data, could use logit function
def logistic(x,w,b):
    """
    x, X data
    w, weights
    b, bias
    """
    return tensor.nnet.softmax(tensor.dot(x,w)+b) #x.dim1=w.dim0

#initialize weights, bias 

w_shp=(3*32*32,10) #image size 3*32*32(IN), 10 category(OUT)
b_shp=10
w=theano.shared(numpy.random.random(w_shp),name='w')
b=theano.shared(numpy.random.random(b_shp),name='b')

#cost function, update rule
yy=logistic(x,w,b)
label_predict=tensor.argmax(yy,axis=1)

alpha=tensor.dscalar('alpha') #regularization term
alpha=0.05

cost=tensor.mean(tensor.nnet.categorical_crossentropy(yy,t))
grad=tensor.grad(cost,[w,b])
lr=0.6
updates=[(w,w-alpha*grad[0]*lr),(b,b-alpha*grad[1]*lr)]


#function
train=theano.function([x,t],cost,updates=updates)
predict=theano.function([x],label_predict)

#initialize some diagnostic variable and train the model
step_cost=100.0
batch_size=164

i=0
while (step_cost >2.0):
      icost=[]
      print "iteration %d " %i
      for batch in range(0,len(X_train),batch_size):
          X_batch=X_train[batch:batch+batch_size]
          Y_batch=Y_train[batch:batch+batch_size]
          icost.append(float(train(X_batch,Y_batch)))
      i=i+1
      step_cost=numpy.mean(icost)
      print "cost %0.8f " %(step_cost)
      label_predict=predict(X_train)
      accuracy=numpy.mean(label_predict==label_train)
      print "Training accuracy %0.8f " %accuracy
