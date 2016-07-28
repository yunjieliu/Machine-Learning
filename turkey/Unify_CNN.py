"""
Theano CNN model for TC, ETC, AR, FR
Here we will need a Spatial Pyramid Pooling layer that pool feature maps to a 
fixed size feature array before feeding into the fully connected layer (Faster RCNN, SPP-net 
has similar implementation)
"""
import os,sys
import theano
from theano import tensor 
from theano.tensor.nnet.abstract_conv import conv2d, get_conv_output_shape
from theano.tensor.signal.pool import pool_2d
import numpy
import math
import  data_load


#*********************************************
#weight initialization
def weight_init(shp,type,name):
    """
    weight initialization
    shp:  shape of the weights
    type: weight initialize type
    name: name of the shared variable
    """
    if type=='uniform':
       print "do uniform weights initialization"
       return theano.shared(numpy.random.uniform(-0.1,0.1,numpy.prod(shp)).reshape(shp),name=name)

    if type=='velocity': #specificly for momentum velocity initialization
       print "momemtum velocity initialization"       
       return theano.shared(numpy.zeros(shp),name=name)


def update_scheme(learn_rate,momentum,decay,parameters,costs,velocity):
    """
    weights update scheme with momentum and weight decay regularization
    learn_rate:  learning rate
    momentum:  momentum coefficient
    decay: weight decay coefficient
    parameters: weights parameters
    costs:  cost function value
    velocity: velocity in momentum
    """      
    grad=tensor.grad(costs,parameters)
    uup=[]
    for i,(p,v) in enumerate(zip(parameters,velocity)):
        uup.append((v,v*momentum-learn_rate*(grad[i]+decay*p)))
        uup.append((p,p+v))

    return uup
    

#Spatial Pyramid Pooling layer
def spp(feature,feature_size,pyramid,pool_type='max'):
    """
    This layer took feature map from previous layer and conduct spatial pyramid pooling,
    return a fixed length feature map/vector (Faster RCNN, SPP-net)
    pyramid: spatial pyramid pooling array e.g. [(3,3),(2,2),(1,1)]
    pool_type:  how to do pooling
    feature: featuer maps from previsou layer  (batch, maps, height, width)
    output: fixed size feature vector 
    """
    assert feature.ndim==4
    height=feature_size[-2]
    width=feature_size[-1]

    #SPP layer essentially is max/average pooling layer, but at different location of the input image
    #Implement 3 level SPP as in the SPP_net paper, might need to asjust this while acrually running model

    if pool_type=='max':
       print "SPP using maxing pooling"
    elif pool_type=='average_exc_pad':
       print "SPP using average pooling (exclude padding)"

    pyramid_height=len(pyramid)
    print "SPP using %d level of pyramid " %pyramid_height 
    
    temp=[]
    for i in range(pyramid_height):
        pyramid_level=pyramid[i] 
        #Assume the pooling kernel will be a square, NxN, can easily change to NxM 
        #here to adjust stride size instead of padding, can easily incoporate padding
        kernel_h=int(math.ceil(float(height)/pyramid_level[0]))
        kernel_w=int(math.ceil(float(width)/pyramid_level[1]))
        stride_h=int(math.floor(float(height)/pyramid_level[0]))
        stride_w=int(math.floor(float(width)/pyramid_level[1]))
        pout=pool_2d(feature,(kernel_h,kernel_w),st=(stride_h,stride_w),mode=pool_type,ignore_border=True) 
        ppp=get_pool_output_shape(feature_size,(kernel_h,kernel_w),(stride_h,stride_w),ignore_border=True)
        print pout 
        poutt=tensor.reshape(pout,(feature_size[:2]+(numpy.prod(pyramid_level),)))
        temp.append(poutt)
        
    return tensor.concatenate((temp[0],temp[1],temp[2]),axis=2)
         
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



#two conv followed by two max pool, then 2 fully connect
def CNN(x,c_l1,c_l2,f_l1,f_l2,PP,ims):
    print ims
    conv1=tensor.nnet.relu(conv2d(x,c_l1)) #default stride=1 --subsample=(1,1) 
    conv1_shp=get_conv_output_shape(ims,c_l1.get_value().shape,border_mode='valid',subsample=(1,1))
    print  conv1_shp
    pp=tensor.reshape(conv1,conv1_shp[:2]+(conv1_shp[2]*conv1_shp[3],))
    print pp 
    pool1=pool_2d(conv1,(2,2),st=(2,2),ignore_border=True)  #default maxpool
    pool1_shp=get_pool_output_shape(conv1_shp,pool_size=(2,2),st=(2,2),ignore_border=True)
    print pool1_shp
    conv2=tensor.nnet.relu(conv2d(pool1,c_l2))
    conv2_shp=get_conv_output_shape(pool1_shp,c_l2.get_value().shape,border_mode='valid',subsample=(1,1))   
    print conv2_shp
    #pool2=pool_2d(conv2,(2,2),st=(2,2),ignore_border=True)
    pool2=spp(conv2,conv2_shp,PP,'max')

    fpool2=tensor.flatten(pool2,outdim=2)

    full1=tensor.nnet.relu(tensor.dot(fpool2,f_l1))
    pyx=tensor.nnet.sigmoid(tensor.dot(full1,f_l2))
    return c_l1, c_l2, f_l1, f_l2, pyx


#********************************************
#misc information
rng_seed=2
data_dir="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/DATA/"
file_name="hurricanes.h5"
data_dict=["1","0"] #group name of positive and negative examples
train_num_p=train_num_n=128*5  #positive and negative training example
valid_num_p=valid_num_n=128*5
test_num_p=test_num_n=128*5     #positive and negative testing example

norm_type= 2    # #1: global contrast norm, 2:standard norm, 3:l1/l2 norm, scikit learn

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
x=tensor.tensor4('x')
t=tensor.matrix('t')

#********************************************
#Initialize weights, bias etc
#NOTE, here need to calculate the dimension of input/output feature size

#First ConvLayer
c_l1_shp=((16,8,5,5)) #number of kernel, input channel, kernel size
c_l1=weight_init(c_l1_shp,'uniform','c_l1')
mc_l1=weight_init(c_l1_shp,'velocity','mc_l1')

#Second ConvLayer
c_l2_shp=((32,16,5,5))
c_l2=weight_init(c_l2_shp,'uniform','c_l2')
mc_l2=weight_init(c_l2_shp,'velocity','mc_l2')

#First Full Layer
f_l1_shp=((32*(3*3+2*2+1*1)),200)
f_l1=weight_init(f_l1_shp,'uniform','f_l1')
mf_l1=weight_init(f_l1_shp,'velocity','mf_l1')

#Second Full Layer
f_l2_shp=(200,2)
f_l2=weight_init(f_l2_shp,'uniform','f_l2')
mf_l2=weight_init(f_l2_shp,'velocity','mf_l2')


#some useful parameters
PP=[(3,3),(2,2),(1,1)]
ims=(128,3,32,32)
#cost and update strategy, learning rule
fcl1,fcl2,ffl1,ffl2,ppyx= CNN(x,c_l1,c_l2,f_l1,f_l2,PP,ims)
label_predict=tensor.argmax(ppyx,axis=1)

cost=tensor.mean(tensor.nnet.binary_crossentropy(ppyx,t))

params=[c_l1,c_l2,f_l1,f_l2]
velocity=[mc_l1,mc_l2,mf_l1,mf_l2]

lr=0.01  #learning rate
mom=0.9 #momentum coefficient
wdecay=0.005  #weight decay

updates=update_scheme(lr,mom,wdecay,params,cost,velocity)

#*************************************************
#function
train=theano.function([x,t],cost,updates=updates)
predict=theano.function([x],label_predict)

#train model
step_cost=100.0
batch_size=128
epoches=20

i=0
while (step_cost >1.0 or i <epoches):
      icost=[]
      print "iteration %d " %i
      for batch in range(0,len(X_train),batch_size):
          X_batch=X_train[batch:batch+batch_size]
          Y_batch=Y_train[batch:batch+batch_size]
          ims=X_batch.shape
          icost.append(float(train(X_batch,Y_batch)))
      i=i+1
      step_cost=numpy.mean(icost)
      print "cost %0.8f " %(step_cost)

      if i%5 ==0:
         label_predict=predict(X_valid[:batch_size])
         accuracy=numpy.mean(label_predict==label_valid[:batch_size])
         print "Validating accuracy %0.8f " %accuracy
         label_predict=predict(X_train[:batch_size])
         accuracy=numpy.mean(label_predict==label_train[:batch_size])
         print "Training accuracy  %0.8f " %accuracy

#the Spatial Pyramid Pooling layer works, the only thing that is wierd about this code is that the training/testing data size has to be interger multiplication
#of batch size. Not generally enough. Might need to redesign the code structure.
#Seems like it is not possible to pass the training data size (batch, channel, width, height) from theano function down the lin from theano function down the linee as an array. Theano function takes tensor.varaible as input, hard to get actual value of variables down the line. There must be other way to do so
