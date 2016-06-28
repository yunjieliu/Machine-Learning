"""
Theano Linear regression
"""
import os
import numpy
#import matplotlib
#from matplotlib import pyplot
import theano
from theano import tensor

#training data
X_train=numpy.linspace(-1,1,101)
Y_train=5*X_train+ 0.7 #+numpy.random.randn(*X_train.shape)*0.2

#plot out training datset
#pyplot.scatter(X_train,Y_train)

#define symbolic in theano
x=tensor.scalar()
t=tensor.scalar()

#linear regresison model
def line_regress(x,w,b):
    """
    x, x input data
    w, weights
    b, bias
    """
    return x*w+b

#claim weights and bias for the fitting model
w=theano.shared(0.0) #weights, coefficient
b=theano.shared(0.0) #bias, intersection
lr=0.005
y=line_regress(x,w,b)

cost=tensor.sqrt(tensor.mean((y-t)**2)) #root mean square cost, Why operate on one element??
grad=tensor.grad(cost,[w,b]) #gradient of cost with respect to weights, and bias

updates=[(w,w-grad[0]*lr),(b,b-grad[1]*lr)]

#compile theano function
train=theano.function([x,t],cost,updates=updates)   


#initialize some diagniostic variable and train model
step_cost=100.0
icost=[]
#train model
i=0
while (step_cost>0.1):
      print "on iteration %d" %i
      for x, t in zip(X_train, Y_train):
          icost.append(float(train(x,t)))  #how to get cost here?
      i=i+1
      step_cost=numpy.mean(icost)
      print "cost %0.8f " %(step_cost)
      print "w= %0.4f, b= %0.4f " %(w.get_value(),b.get_value())
    
