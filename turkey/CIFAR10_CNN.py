"""
CIFAR10 CNN model
"""
import os
import theano
from theano import tensor, printing
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
import numpy
import load

#get training, testing data
#training, testing data
X_train, Y_train, X_test, Y_test= load.cifar10(dtype=theano.config.floatX)

#data are flatten into array, reshape them back 
X_train=X_train.reshape(-1,3,32,32)
X_test=X_test.reshape(-1,3,32,32)

label_train=numpy.argmax(Y_train,axis=1)

#define symbolic in theano
x=tensor.tensor4()
t=tensor.matrix()


#CNN model

def CNN(x,c_l1,c_l2,f_l1,f_l2):
    conv1=tensor.nnet.relu(conv2d(x,c_l1)) #default stride=1 --subsample=(1,1) 
    pool1=pool_2d(conv1,(2,2),st=(2,2),ignore_border=True)  #default maxpool
    conv2=tensor.nnet.relu(conv2d(pool1,c_l2))
    pool2=pool_2d(conv2,(2,2),st=(2,2),ignore_border=True)
    fpool2=tensor.flatten(pool2,outdim=2)
    full1=tensor.nnet.relu(tensor.dot(fpool2,f_l1))
    pyx=tensor.nnet.softmax(tensor.dot(full1,f_l2))
     
    return c_l1, c_l2, f_l1, f_l2, pyx

#initialize weights, bias
#NOTE, here need to calculate the dimension of input/output feature size
c_l1_shp=((16,3,5,5)) #number of kernel, input channel, kernel size
c_l1=theano.shared(numpy.random.uniform(-0.1,0.1,numpy.prod(c_l1_shp)).reshape(c_l1_shp),name='c_l1')
mc_l1=theano.shared(numpy.zeros(c_l1_shp),name='mc_l1')
c_l2_shp=((32,16,5,5))
c_l2=theano.shared(numpy.random.uniform(-0.1,0.1,numpy.prod(c_l2_shp)).reshape(c_l2_shp),name='c_l2')
mc_l2=theano.shared(numpy.zeros(c_l2_shp),name='mc_l2')
f_l1_shp=((32*5*5),500)
f_l1=theano.shared(numpy.random.uniform(-0.1,0.1,numpy.prod(f_l1_shp)).reshape(f_l1_shp),name='f_l1')
mf_l1=theano.shared(numpy.zeros(f_l1_shp),name='mf_l1')
f_l2_shp=(500,10)
f_l2=theano.shared(numpy.random.uniform(-0.1,0.1,numpy.prod(f_l2_shp)).reshape(f_l2_shp),name='f_l2')
mf_l2=theano.shared(numpy.zeros(f_l2_shp),name='mf_l2')

#cost and update strategy, learning rule
fcl1,fcl2,ffl1,ffl2,ppyx= CNN(x,c_l1,c_l2,f_l1,f_l2)
label_predict=tensor.argmax(ppyx,axis=1)

cost=tensor.mean(tensor.nnet.categorical_crossentropy(ppyx,t))
grad=tensor.grad(cost,[c_l1,c_l2,f_l1,f_l2])
lr=0.01
momentum=0.9

updates=[(mc_l1,mc_l1*momentum-lr*grad[0]),(c_l1,c_l1+mc_l1),
(mc_l2,mc_l2*momentum-lr*grad[1]),(c_l2,c_l2+mc_l2),
(mf_l1,mf_l1*momentum-lr*grad[2]),(f_l1,f_l1+mf_l1),
(mf_l2,mf_l2*momentum-lr*grad[3]),(f_l2,f_l2+mf_l2)]

#function
train=theano.function([x,t],cost,updates=updates)
predict=theano.function([x],label_predict)


#train model
step_cost=100.0
batch_size=128

i=0
while (step_cost >1.0):
      icost=[]
      print "iteration %d " %i
      for batch in range(0,len(X_train),batch_size):
          X_batch=X_train[batch:batch+batch_size]
          Y_batch=Y_train[batch:batch+batch_size]
          icost.append(float(train(X_batch,Y_batch)))
      i=i+1
      step_cost=numpy.mean(icost)
      print "cost %0.8f " %(step_cost)
      label_predict=predict(X_train[:2000])
      accuracy=numpy.mean(label_predict==label_train[:2000])
      print "Training accuracy %0.8f " %accuracy
