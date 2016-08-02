"""
Lasagne CNN for TC
"""
import theano
from theano import tensor
import lasagne
import numpy
import data_load



def CNN(input_var=None):
    """
    Build ConvNet on lasagne tool box, layer structure similar to the one build on Theano for unified network
    1) use 3D convolution as the first convolution layer to deal with different channel inputs
    2) use Spatial Pyramid Pooling layer before fully connected layer to deal with different dimension input 
    """

    convnet=lasagne.layers.InputLayer(shape=(None,8,32,32),input_var=input_var,name='inputlayer')

    convnet=lasagne.layers.Conv2DLayer(convnet,num_filters=16, filter_size=(5, 5), stride=(1, 1), pad=0, \
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),name='conv1')

    convnet=lasagne.layers.MaxPool2DLayer(convnet,pool_size=(2, 2),stride=None, pad=(0, 0), ignore_border=True,name='pool1')

    convnet=lasagne.layers.Conv2DLayer(convnet,num_filters=32, filter_size=(5, 5), stride=(1, 1), pad=0, \
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),name='conv2')

    convnet=lasagne.layers.MaxPool2DLayer(convnet,pool_size=(2, 2),name='pool2')

    convnet=lasagne.layers.DenseLayer(convnet,num_units=500,nonlinearity=lasagne.nonlinearities.rectify,name='full1')

    convnet=lasagne.layers.DenseLayer(convnet,num_units=2,nonlinearity=lasagne.nonlinearities.sigmoid,name='full2')

    return convnet


#get training data
rng_seed=2
numpy.random.seed(rng_seed)
data_dir="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/DATA/"
file_name="hurricanes.h5"
data_dict=["1","0"] #group name of positive and negative examples
train_num_p=train_num_n=8000  #positive and negative training example
valid_num_p=valid_num_n=1000
test_num_p=test_num_n=1000    #positive and negative testing example

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

#build CNN
convnet=CNN(input_var)

print "Great, CNN build...."

#loss function, paramaters, updates (learning rules)
prediction = lasagne.layers.get_output(convnet)
label_predict=tensor.argmax(prediction,axis=1)

loss= lasagne.objectives.binary_crossentropy(prediction, target_var)
loss=loss.mean() 

params=lasagne.layers.get_all_params(convnet,trainable=True)
updates=lasagne.updates.momentum(loss,params,learning_rate=0.01,momentum=0.9)

print "Great,gradient, loss, updates build..."
#train function
train=theano.function([input_var,target_var],loss,updates=updates)
predict=theano.function([input_var],label_predict)

print "Great, function build...."
#training the model
print "starts model training...."

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
          train_loss += float(train(X_batch,Y_batch))
          train_batch += 1

      print "training loss: %0.8f " %float(train_loss/train_batch) 
      epoch=epoch+1

      if epoch %5 ==0: #then we print out training accuracy
         label_predict=predict(X_train)
         accuracy=numpy.mean(label_predict==label_train)
         print "Validating accuracy %0.8f " %accuracy

 
    

