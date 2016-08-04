"""
Theano CNN model for AR
"""
import os,sys
import theano
from theano import tensor 
from theano.tensor.nnet.abstract_conv import conv2d, get_conv_output_shape
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
train_num_p=train_num_n= 10  #positive and negative training example
valid_num_p=10
valid_num_n=10
test_num_p=test_num_n=10     #positive and negative testing example

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
def get_pool_output_shape(mapsize,pool_size,st,ignore_border):
    """
    calculate feature map shape after pooling operation
    mapsize: input feature map size (batchsize,feature map, height, width)
    pool_size: pooling window
    st: pooling stride
    ignore_border: same as pool_2d
    """
    assert len(mapsize) ==4  #(batchsize,maps,height,weight)
    if ignore_border:
       out=((mapsize[-2]-pool_size[0])/st[0]+1,(mapsize[-1]-pool_size[1])/st[1]+1)
    else:
       out=((mapsize[-2]-pool_size[0])/st[0]+2,(mapsize[-1]-pool_size[1])/st[1]+2)

    return (mapsize[:2]+out)


def LRN(feature,feature_size,alpha=0.001,k=1,beta=0.75,n=5):
    """
    Local Response Normalization (AlexNet)
    Neon, lasagne and Caffe has similar layer implementations 
    Here we are going to take Lasagne implementation as reference
    https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/normalization.py#L55-L117
    formula xi=xi/[(k+alpha*sum(xj^2))^beta], sum(xj^2) from i-n/2 to i+n/2 , e.g. if i==0, sum over
            0 to n feature maps, if i==5, sum over i-n/2 to i+n/2

    feature: input feature maps
    alpha: scaling factor
    k: offsets
    beta: power factor
    n: cross map normal size, better be odd
    """
    assert feature.ndim ==4  #batch size, channel, width, height

    bsize,chan,w,h=feature_size
    half_n=n // 2 #"//" do the interger division
    feature_sqr=tensor.sqr(feature)
    print "LRN input size ",feature_size

    #Lasagne way of doing this
    extra_channels = tensor.alloc(0., bsize, chan + 2*half_n, w, h)
    feature_sqr = tensor.set_subtensor(extra_channels[:, half_n:half_n+chan, :, :],feature_sqr)
    for i in range(n):
        k=k+alpha * feature_sqr[:, i:i+chan, :, :] 
    k=k**beta
    
    return feature/k
    
    #alternatively we can do following   
    #norm_feature=tensor.zeros_like(feature)
    #for i in range(chan):
    #    term=k+alpha*tensor.sum(feature_sqr[:,max(i-half_n,0):min(chan-1,i+half_n),:,:],axis=1)
    #    term_pow=tensor.pow(term,beta)
    #    norm_feature=tensor.set_subtensor(norm_feature[:,i,:,:],feature[:,i,:,:]/term_pow)
 
    #return norm_feature   


#build the CNN model
#two conv followed by two max pool, then 2 fully connect
def CNN(x,c_l1,c_l2,f_l1,f_l2,insize):
    print "in size ", insize
    conv1=tensor.nnet.relu(conv2d(x,c_l1)) #default stride=1 --subsample=(1,1) 
    conv1_shp=get_conv_output_shape(insize,c_l1.get_value().shape,border_mode='valid',subsample=(1,1))
    print "conv1 size ", conv1_shp
    pool1=pool_2d(conv1,(3,3),st=(3,3),ignore_border=True)  #default maxpool
    pool1_shp=get_pool_output_shape(conv1_shp,pool_size=(3,3),st=(3,3),ignore_border=True)
    print "pool1 size ", pool1_shp
    lrn1=LRN(pool1,pool1_shp)
    lrn1_shp=tuple(pool1_shp)
    print "cross map norm1 size ", lrn1_shp
    conv2=tensor.nnet.relu(conv2d(lrn1,c_l2))
    conv2_shp=get_conv_output_shape(lrn1_shp,c_l2.get_value().shape,border_mode='valid',subsample=(1,1))
    print "conv2 size ", conv2_shp 
    pool2=pool_2d(conv2,(2,2),st=(2,2),ignore_border=True)
    pool2_shp=get_pool_output_shape(conv2_shp,pool_size=(2,2),st=(2,2),ignore_border=True)
    print "pool2 size ", pool2_shp
    lrn2=LRN(pool2,pool2_shp)
    lrn2_shp=tuple(pool2_shp)
    print "cross map norm2 size " , lrn2_shp
    fpool2=tensor.flatten(lrn2,outdim=2)

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
f_l1_shp=((16*17*30),200)
f_l1=theano.shared(numpy.random.uniform(-0.1,0.1,numpy.prod(f_l1_shp)).reshape(f_l1_shp),name='f_l1')
mf_l1=theano.shared(numpy.zeros(f_l1_shp),name='mf_l1')
f_l2_shp=(200,2)
f_l2=theano.shared(numpy.random.uniform(-0.1,0.1,numpy.prod(f_l2_shp)).reshape(f_l2_shp),name='f_l2')
mf_l2=theano.shared(numpy.zeros(f_l2_shp),name='mf_l2')

#cost and update strategy, learning rule
lr=0.03  #learning rate
momentum=0.9 #momentum coefficient
wdecay=0.001  #weight decay
batch_size=10
image_size=(2,148,224)
input_size=tuple(numpy.append(batch_size,image_size))


fcl1,fcl2,ffl1,ffl2,ppyx= CNN(x,c_l1,c_l2,f_l1,f_l2,input_size)
label_predict=tensor.argmax(ppyx,axis=1)

cost=tensor.mean(tensor.nnet.binary_crossentropy(ppyx,t))
grad=tensor.grad(cost,[c_l1,c_l2,f_l1,f_l2])

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

      if i%10 ==0:
         label_predict=[]
         for batch in range(0,len(X_valid),batch_size):
             X_batch=X_valid[batch:batch+batch_size]
             label_predict.extend(predict(X_batch))
         accuracy=numpy.mean(label_predict==label_valid)
         logger.info("Validating accuracy %0.8f " %accuracy)

         #label_predict=predict(X_train)
         #logger.info("thus far")
         #accuracy=numpy.mean(label_predict==label_train)
         #logger.info("Training accuracy  %0.8f " %accuracy)
