"""
Lasagne CNN for TC
"""
import theano
from theano import tensor
import lasagne
import numpy
import data_load



def CNN(input_var):
    """
    Build ConvNet on lasagne tool box, layer structure similar to the one build on Theano for unified network
    1) use 3D convolution as the first convolution layer to deal with different channel inputs
    2) use Spatial Pyramid Pooling layer before fully connected layer to deal with different dimension input 
    """
    In=lasagne.layers.InputLayer(shape=(None,2,148,224),input_var=input_var,name='inputlayer')

    Conv1=lasagne.layers.Conv2DLayer(In,num_filters=16, filter_size=(5, 5), stride=(1, 1), pad=0, \
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),name='conv1')

    Pool1=lasagne.layers.MaxPool2DLayer(Conv1,pool_size=(2, 2),stride=None, pad=(0, 0), ignore_border=True,name='pool1')

    Lrn1=lasagne.layers.LocalResponseNormalization2DLayer(Pool1,alpha=0.001,k=1,beta=0.75,n=5)

    Conv2=lasagne.layers.Conv2DLayer(Lrn1,num_filters=32, filter_size=(5, 5), stride=(1, 1), pad=0, \
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),name='conv2')

    Pool2=lasagne.layers.MaxPool2DLayer(Conv2,pool_size=(2, 2),name='pool2')

    Lrn2=lasagne.layers.LocalResponseNormalization2DLayer(Pool2,alpha=0.001,k=1,beta=0.75,n=5)

    Full1=lasagne.layers.DenseLayer(Lrn2,num_units=500,nonlinearity=lasagne.nonlinearities.rectify,name='full1')

    Full2=lasagne.layers.DenseLayer(Full1,num_units=2,nonlinearity=lasagne.nonlinearities.sigmoid,name='full2')

    return Full2 #each layer kined to its inputs, so just need output layer to get information from all layers


#get training data
rng_seed=2
numpy.random.seed(rng_seed)
data_dir="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/DATA/"
file_name="hurricanes.h5"
data_dict=["1","0"] #group name of positive and negative examples
train_num_p=train_num_n=128  #positive and negative training example
valid_num_p=valid_num_n=128
test_num_p=test_num_n=128   #positive and negative testing example

norm_type= 3    # #1: global contrast norm, 2:standard norm, 3:l1/l2 norm, scikit learn

#get training , testing data
#hurricane image size 8x32x32
(X_train, Y_train), (X_valid, Y_valid), (X_test,Y_test)= \
          data_load.happy_loader(rng_seed,data_dir,file_name,
                                 data_dict,
                                 train_num_p,valid_num_p,test_num_p,
                                 train_num_n,valid_num_n,test_num_n,
                                 norm_type,normalize=True)
label_train=numpy.argmax(Y_train,axis=1)
#theano variables
input_var = tensor.tensor4('inputs')
target_var = tensor.matrix('targets')
size=tensor.vector('size')

#build CNN
convnet=CNN(input_var)

#loss function, paramaters, updates (learning rules)
prediction = lasagne.layers.get_output(convnet)
label_predict=tensor.argmax(prediction,axis=1)

loss= lasagne.objectives.binary_crossentropy(prediction, target_var)
loss=loss.mean() 

params=lasagne.layers.get_all_params(convnet,trainable=True)
updates=lasagne.updates.momentum(loss,params,learning_rate=0.01,momentum=0.9)

#train function
train=theano.function([input_var,target_var],loss,updates=updates)
predict=theano.function([input_var],label_predict)

#training the model

num_epochs=1000
epoch=0
batch_size=128
train_loss=0
train_batch=0

while epoch<num_epochs:
      print "on epoch... %d " %epoch
      for batch in range(0,len(X_train),batch_size):
          X_batch=X_train[batch:batch+batch_size]
          Y_batch=Y_train[batch:batch+batch_size]
          bshape=X_batch.shape
          train_loss += float(train(X_batch,Y_batch))
          train_batch += 1

      print "training loss: %0.8f " %float(train_loss/train_batch) 
      epoch=epoch+1

      if epoch %5 ==0: #then we print out training accuracy
         label_predict=predict(X_train)
         accuracy=numpy.mean(label_predict==label_train)
         print "Validating accuracy %0.8f " %accuracy

      #if epoch %10 ==0:  #then we print out validating accuracy
      #   label_predict =predict(X_valid) 
      #   accuracy=numpy.mean(label_predict==label_valid)