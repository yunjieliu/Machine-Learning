"""
Theano CNN model for TC, ETC, AR, FR
Here we will need a Spatial Pyramid Pooling layer that pool feature maps to a 
fixed size feature array before feeding into the fully connected layer (Faster RCNN, SPP-net 
has similar implementation)
"""
import os,sys
import theano
from theano import tensor 
from theano.tensor.nnet import conv3D
from theano.tensor.nnet.abstract_conv import conv2d, get_conv_output_shape
from theano.tensor.signal.pool import pool_2d
import numpy
import math
import  data_load_all


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
        print ppp 
        print feature_size[:2]+(numpy.prod(pyramid_level),) 
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
    #-------
    #conv3D get rid of dependency of the number of input image channel
    b=numpy.zeros(c_l1.get_value().shape[0])
    conv1=tensor.nnet.relu(conv3D(x.dimshuffle(0,2,3,1,'x'),c_l1.dimshuffle(0,2,3,1,'x'),b,d=(1,1,1))) # shuffle dimensions
    conv1=tensor.sum(conv1,axis=3) #add the dimension of channels
    conv1=conv1.dimshuffle(0,3,1,2) #shuffle back to same dimension as conv2D
    #---------

    #conv1=tensor.nnet.relu(conv2d(x,c_l1)) #default stride=1 --subsample=(1,1) 
    conv1_shp=get_conv_output_shape(ims,c_l1.get_value().shape,border_mode='valid',subsample=(1,1))
    print  conv1_shp

    #pp=tensor.reshape(conv1,conv1_shp[:2]+(conv1_shp[2]*conv1_shp[3],))
    #print pp 

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
    pyx=tensor.nnet.softmax(tensor.dot(full1,f_l2))
    return c_l1, c_l2, f_l1, f_l2, pyx


#********************************************
#misc information
rng_seed=2
file_name1="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/DATA/hurricanes.h5"
file_name2="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/DATA/atmospheric_river_us+eu+landsea_sep10.h5"
file_name3="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/DATA/fronts_all.h5"


train_num_p=128*1 #positive and negative training example
valid_num_p=128*1
test_num_p=128*1     #positive and negative testing example
classes=3   #types of objects
norm_type= 3    # #1: global contrast norm, 2:standard norm, 3:l1/l2 norm, scikit learn

#*********************************************
#get training , testing data
#hurricane image size 8x32x32
groups=['1','0']
now_class=0
(Hx_train, Hy_train), (Hx_valid, Hy_valid), (Hx_test,Hy_test)= \
          data_load_all.happy_loader(rng_seed,file_name1,groups,
                                 train_num_p,valid_num_p,test_num_p,
                                 classes,now_class,norm_type,normalize=True)

groups=['AR','Non_AR']
now_class=1
(Ax_train, Ay_train), (Ax_valid, Ay_valid), (Ax_test,Ay_test)= \
          data_load_all.happy_loader(rng_seed,file_name2,groups,
                                 train_num_p,valid_num_p,test_num_p,
                                 classes,now_class,norm_type,normalize=True)


groups=['Front','NonFront']
now_class=2
(Fx_train, Fy_train), (Fx_valid, Fy_valid), (Fx_test,Fy_test)= \
          data_load_all.happy_loader(rng_seed,file_name3,groups,
                                 train_num_p,valid_num_p,test_num_p,
                                 classes,now_class,norm_type,normalize=True)

assert Hx_train.ndim ==4 and  Hy_train.ndim ==2

#validating ground truth label
#label_train=numpy.argmax(Y_train,axis=1)
#label_valid=numpy.argmax(Y_valid,axis=1)
#label_test=numpy.argmax(Y_test,axis=1)
H_valid=numpy.argmax(Hy_valid,axis=1)
A_valid=numpy.argmax(Ay_valid,axis=1)
F_valid=numpy.argmax(Fy_valid,axis=1)

#define symbolic in theano
x=tensor.tensor4('x')
t=tensor.matrix('t')

#********************************************
#Initialize weights, bias etc
#NOTE, here need to calculate the dimension of input/output feature size

#First ConvLayer
c_l1_shp=((16,1,5,5)) #number of kernel, input channel, kernel size
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
f_l2_shp=(200,3)
f_l2=weight_init(f_l2_shp,'uniform','f_l2')
mf_l2=weight_init(f_l2_shp,'velocity','mf_l2')


#some useful parameters
PP=[(1,1),(2,2),(3,3)] 
ims=(128,8,32,32)

#cost and update strategy, learning rule
fcl1,fcl2,ffl1,ffl2,ppyx= CNN(x,c_l1,c_l2,f_l1,f_l2,PP,ims)
label_predict=tensor.argmax(ppyx,axis=1)

cost=tensor.mean(tensor.nnet.categorical_crossentropy(ppyx,t))

params=[c_l1,c_l2,f_l1,f_l2]
velocity=[mc_l1,mc_l2,mf_l1,mf_l2]

lr=0.01  #learning rate
mom=0.9 #momentum coefficient
wdecay=0.005  #weight decay

updates=update_scheme(lr,mom,wdecay,params,cost,velocity)

#*************************************************
#function
ims=tensor.matrix('ims')
train=theano.function([x,t],cost,updates=updates)
predict=theano.function([x],label_predict)

#train model
step_cost=100.0
batch_size=128
epoches=10

#To test does SPP layer can unify the networks for different input size:
#Train batch by batch: TC--> AR--> Front --> TC --> AR.... loop till train data end
#To train model using different image size still requires the images within a batch are 
#at the same dimension (I believe so, because these iamges are going to be flatten before
#feeding to model. If these iamges are of different size, how to make sure each time the 
#network are operating on single image rather than cross over?? How the batch learning was done,image by image or 
#a huge matrix?)

#validating ground truth
H_valid=numpy.argmax(Hy_valid,axis=1)
A_valid=numpy.argmax(Ay_valid,axis=1)
F_valid=numpy.argmax(Fy_valid,axis=1)

H_minibatch=len(Hx_train)/batch_size
A_minibatch=len(Ax_train)/batch_size
F_minibatch=len(Fx_train)/batch_size



i=0
H=0
A=0
F=0

while (step_cost >1.0 or i <epoches):
      icost=[]
      print "iteration %d " %i

      for bb in range(max(H_minibatch,A_minibatch,F_minibatch)):
          if bb <=H_minibatch:
             X_batch=Hx_train[bb*batch_size:(bb+1)*batch_size]
             Y_batch=Hy_train[bb*batch_size:(bb+1)*batch_size]
             iii=X_batch.shape
             icost.append(float(train(X_batch,Y_batch)))
             print 'done 1'
          if bb<=A_minibatch:
             X_batch=Ax_train[bb*batch_size:(bb+1)*batch_size]
             Y_batch=Ay_train[bb*batch_size:(bb+1)*batch_size]
             icost.append(float(train(X_batch,Y_batch)))
             print "done 2"
          if bb<=F_minibatch:
             X_batch=Fx_train[bb*batch_size:(bb+1)*batch_size]
             Y_batch=Fy_train[bb*batch_size:(bb+1)*batch_size]
             icost.append(float(train(X_batch,Y_batch)))
      i=i+1
      step_cost=numpy.mean(icost)
      print "cost %0.8f " %(step_cost)

      if i%10 ==0:
         label_predict=predict(X_valid[:batch_size])
         accuracy=numpy.mean(label_predict==label_valid[:batch_size])
         print "Validating accuracy %0.8f " %accuracy
         label_predict=predict(X_train[:batch_size])
         accuracy=numpy.mean(label_predict==label_train[:batch_size])
         print "Training accuracy  %0.8f " %accuracy

#the Spatial Pyramid Pooling layer works, the only thing that is wierd about this code is that the training/testing data size has to be interger multiplication
#of batch size. Not generally enough. Might need to redesign the code structure.

