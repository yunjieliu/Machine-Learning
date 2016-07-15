"""
Theano CNN model for AR
"""
import os,sys
import theano
from theano import tensor 
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
import numpy
import  data_load

import logging
logger=logging.getLogger('model')
handler=logging.StreamHandler()
formatter=logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel("INFO")

#********************************************
#misc information
rng_seed=2
numpy.random.seed(rng_seed)
data_dir="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/DATA/"
file_name="atmospheric_river_us+eu+landsea_sep10.h5"
data_dict=["AR","Non_AR"] #group name of positive and negative examples
train_num_p=train_num_n= 5000  #positive and negative training example
valid_num_p=800
valid_num_n=500
test_num_p=test_num_n=1000     #positive and negative testing example

norm_type= 3   # #1: global contrast norm, 2:standard norm, 3:l1/l2 norm, scikit learn

#*********************************************
#get training , testing data
#hurricane image size 8x32x32
(X_train, Y_train), (X_valid, Y_valid), (X_test,Y_test)= \
          data_load.happy_loader(rng_seed,data_dir,file_name,
                                 data_dict,
                                 train_num_p,valid_num_p,test_num_p,
                                 train_num_n,valid_num_n,test_num_n,
                                 norm_type,normalize=True)

assert X_train.ndim ==4 and  Y_train.ndim ==2

#training, testing ground truth label
label_train=numpy.argmax(Y_train,axis=1)
label_valid=numpy.argmax(Y_valid,axis=1)
label_test=numpy.argmax(Y_test,axis=1)

#define symbolic in theano
x=tensor.tensor4()
t=tensor.matrix()


#*********************************************
#build the CNN model
#two conv followed by two max pool, then 2 fully connect
def CNN(x,c_l1,c_l2,f_l1,f_l2):
    conv1=tensor.nnet.relu(conv2d(x,c_l1)) #default stride=1 --subsample=(1,1) 
    pool1=pool_2d(conv1,(3,3),st=(3,3),ignore_border=True)  #default maxpool
    conv2=tensor.nnet.relu(conv2d(pool1,c_l2))
    pool2=pool_2d(conv2,(3,3),st=(3,3),ignore_border=True)
    fpool2=tensor.flatten(pool2,outdim=2)
    full1=tensor.nnet.relu(tensor.dot(fpool2,f_l1))
    pyx=tensor.nnet.sigmoid(tensor.dot(full1,f_l2))

    return c_l1, c_l2, f_l1, f_l2, pyx


#********************************************
#Initialize weights, bias etc
#NOTE, here need to calculate the dimension of input/output feature size
c_l1_shp=((8,2,12,12)) #number of kernel, input channel, kernel size
c_l1=theano.shared(numpy.random.uniform(-0.1,0.1,numpy.prod(c_l1_shp)).reshape(c_l1_shp),name='c_l1')
mc_l1=theano.shared(numpy.zeros(c_l1_shp),name='mc_l1') #velocity initialize 0
c_l2_shp=((16,8,12,12))
c_l2=theano.shared(numpy.random.uniform(-0.1,0.1,numpy.prod(c_l2_shp)).reshape(c_l2_shp),name='c_l2')
mc_l2=theano.shared(numpy.zeros(c_l2_shp),name='mc_l2')
f_l1_shp=((16*11*20),200)
f_l1=theano.shared(numpy.random.uniform(-0.1,0.1,numpy.prod(f_l1_shp)).reshape(f_l1_shp),name='f_l1')
mf_l1=theano.shared(numpy.zeros(f_l1_shp),name='mf_l1')
f_l2_shp=(200,2)
f_l2=theano.shared(numpy.random.uniform(-0.1,0.1,numpy.prod(f_l2_shp)).reshape(f_l2_shp),name='f_l2')
mf_l2=theano.shared(numpy.zeros(f_l2_shp),name='mf_l2')

#cost and update strategy, learning rule
fcl1,fcl2,ffl1,ffl2,ppyx= CNN(x,c_l1,c_l2,f_l1,f_l2)
label_predict=tensor.argmax(ppyx,axis=1)

cost=tensor.mean(tensor.nnet.binary_crossentropy(ppyx,t))
grad=tensor.grad(cost,[c_l1,c_l2,f_l1,f_l2])

lr=0.03  #learning rate
momentum=0.9 #momentum coefficient
wdecay=0.001  #weight decay

updates=[(mc_l1,mc_l1*momentum-lr*(grad[0]+wdecay*c_l1)),(c_l1,c_l1+mc_l1),
(mc_l2,mc_l2*momentum-lr*(grad[1]+wdecay*c_l2)),(c_l2,c_l2+mc_l2),
(mf_l1,mf_l1*momentum-lr*(grad[2]+wdecay*f_l1)),(f_l1,f_l1+mf_l1),
(mf_l2,mf_l2*momentum-lr*(grad[3]+wdecay*f_l2)),(f_l2,f_l2+mf_l2)]


#*************************************************
#function
train=theano.function([x,t],cost,updates=updates)
predict=theano.function([x],label_predict)


#train model
step_cost=100.0
batch_size=100
epoches=50

i=0
while (step_cost >0.2 or i <epoches):
      icost=[]
      logger.info( "iteration %d " %i)
      for batch in range(0,len(X_train),batch_size):
          X_batch=X_train[batch:batch+batch_size]
          Y_batch=Y_train[batch:batch+batch_size]
          icost.append(float(train(X_batch,Y_batch)))
  
      i=i+1
      step_cost=numpy.mean(icost)
      logger.info("cost %0.8f " %(step_cost))

      if i%1 ==0:
         label_predict=predict(X_valid)
         accuracy=numpy.mean(label_predict==label_valid)
         logger.info("Validating accuracy %0.8f " %accuracy)
         label_predict=predict(X_train)
         logger.info("thus far")
         accuracy=numpy.mean(label_predict==label_train)
         logger.info("Training accuracy  %0.8f " %accuracy)
