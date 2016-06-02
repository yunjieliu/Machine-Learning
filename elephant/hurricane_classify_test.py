#! /usr/bin/env python

"""
Hurricane events classification NEON model
Training and validating model
"""

#lots of module to import
import numpy, ipdb, neon,datetime
import data_load  #personalized data load module
from neon.data import DataIterator
from neon.util.argparser import NeonArgparser
from neon.initializers import Uniform, Constant
from neon.callbacks.callbacks import Callbacks, MetricCallback, LossCallback
from neon.layers import GeneralizedCost, Affine, Conv,Pooling
from neon.transforms import CrossEntropyMulti, CrossEntropyBinary, Misclassification
from neon.transforms import Rectlin, Softmax, Logistic, Identity
from neon.models import Model
from neon.optimizers import GradientDescentMomentum

import logging
logger =logging.getLogger("Model")
logger.setLevel("INFO")

c_time=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

"""
logging information and command parsing
Can specify these argument on command line or here
"""

parser = NeonArgparser(__doc__)
argss=[ "--file_name","--train_num_p","--valid_num_p","--test_num_p",
        "--train_num_n","--valid_num_n","--test_num_n","--nclass",
       #data dependent arguments
       "--data_dict","--norm_type",
       #output dependent arguments
       "--out_dir"]
       
for a in argss:
    parser.add_argument(a)

parser.set_defaults(

       #constant arguments
       rng_seed=2,
       backend= "cpu",
       progress_bar=True,
       verbose=4,        
       evaluation_freq=2,

       #data
       epochs= 1,
       batch_size=100, ####when testing, make the batch size equal to test data length (easier for later confusion matrix and feature sample)

       data_dir="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/DATA/",
       file_name="hurricanes.h5",
       nclass=2,
       data_dict=["1","0"],
       norm_type=2, #1: global contrast norm, 2:standard norm, 3:l1/l2 norm, scikit learn
           

       train_num_p=8000,
       valid_num_p=1000,
       test_num_p=1000,
       train_num_n=8000,
       valid_num_n=1000,
       test_num_n=1000,


       # results
       #out_dir="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/RESULTS/",
       #save_path="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/RESULTS/hurricane_classify_train_S.pkl",
       #serialize= 2,
       logfile="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/RESULTS/hurricane/hurricane_classify_test."+c_time+".log",
       #output_file="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/RESULTS/hurricane_classify_S."+c_time+".h5",
       model_file="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/RESULTS/hurricane/hurricane_classify_train.pkl",
) 

args = parser.parse_args()

"""
data loading and sort training, validating and testing
"""

##here the X_train, X_valid, X_test should be a feature vector e.g (3x28x28 will be 2352)
(X_train, Y_train),(X_valid, Y_valid),(X_test,Y_test)= \
         data_load.happy_loader(args.data_dir,args.file_name,
                                  args.data_dict,
                                  args.train_num_p,args.valid_num_p,args.test_num_p,
                                  args.train_num_n,args.valid_num_n,args.test_num_n,
                                  args.norm_type,normalize=True)

train=DataIterator(X_train,Y_train,nclass=args.nclass,lshape=(8,32,32))
valid=DataIterator(X_valid,Y_valid,nclass=args.nclass,lshape=(8,32,32))
test=DataIterator(X_test,Y_test,nclass=args.nclass,lshape=(8,32,32))

logger.info("load data complete...")
"""
default back end gernate will be CPU, here pass that section
"""

"""
construct the model
"""
#initialize weights

init_uni = Uniform(low=-0.1, high=0.1)

#learning rule

opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                  momentum_coef=0.9,
                                  stochastic_round=args.rounding,
                                  wdecay=0.005 )

#model layers and its activation

layers=[]
layers.append(Conv((5, 5, 8), init=init_uni, activation=Rectlin(),batch_norm=False))
layers.append(Pooling((2, 2),strides=2))
layers.append(Conv((5, 5, 16), init=init_uni,activation=Rectlin(), batch_norm=False))
layers.append(Pooling((2, 2),strides=2))
layers.append(Affine(nout=50, init=init_uni,activation=Rectlin(), batch_norm=False))
layers.append(Affine(nout=2, init=init_uni, activation=Logistic()))

#cost function

cost = GeneralizedCost(costfunc=CrossEntropyBinary())

#final model

mlp = Model(layers=layers)

logger.info("model construction complete...")

"""
model training and classification accurate rate
"""
#load trained weights
if args.model_file:
   mlp.load_weights(args.model_file)


#model training and results
callbacks = Callbacks(mlp,train, args, eval_set=test,metric=Misclassification())

#run the model

mlp.fit(train, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

#final classification accuracy

#t_mis_rate=mlp.eval(train, metric=Misclassification())*100
#v_mis_rate=mlp.eval(valid, metric=Misclassification())*100
test_mis_rate=mlp.eval(test, metric=Misclassification())*100

#print ('Train Misclassification error = %.1f%%' %t_mis_rate)
#print ('Valid Miscladdifcaiton error = %.1f%%' %v_mis_rate)
print ('Test Miscladdifcaiton error = %.1f%%' %test_mis_rate)

"""
filters and features learned by the model
"""
def vis_feature(layers):
    import matplotlib
    from matplotlib import pyplot

    first_convlayer=layers[0]
    second_convlayer=layers[2]
    
    input_image=first_convlayer[0].inputs
    first_convlayer_feature=first_convlayer[0].outputs
    first_convlayer_activation=first_convlayer[1].outputs
    
    second_convlayer_feature=second_convlayer[0].outputs
    second_convlayer_activation=second_convlayer[1].outputs
    
    input_image=input_image.asnumpyarray()
    first_convlayer_feature=first_convlayer_feature.asnumpyarray()
    second_convlayer_feature=second_convlayer_feature.asnumpyarray()

    input_image=input_image.reshape(8,32,32,100)
    first_convlayer_feature=first_convlayer_feature.reshape(8,28,28,100)
    second_convlayer_feature=second_convlayer_feature.reshape(16,10,10,100)
    for jj in range(20): 
        logger.info("on example...%d" %jj)
        #input iamge
        vars=["TMQ","V850","PSL","U850","T500","UBOT","T200","VBOT"]
        fig,axis=pyplot.subplots(ncols=8,nrows=1,figsize=(8,3))
        pyplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.01, hspace=0.001)
        for (i,ax),var in zip(enumerate(axis.ravel()),vars):
            ax.imshow(input_image[i,:,:,jj])
            ax.tick_params(axis="x",bottom="off",top="off",labelbottom="off")
            ax.tick_params(axis="y",left="off",right="off",labelleft="off")
            ax.set_title(var,fontsize=12)
        pyplot.tight_layout()
        #pyplot.show()
        pyplot.savefig("input_image_"+str(jj)+".png")

  
        #first convolution features
        fig,axis=pyplot.subplots(ncols=8,nrows=1,figsize=(8,3))
        pyplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.01, hspace=0.001) 
        for i,ax in enumerate(axis.ravel()):
            ax.imshow(first_convlayer_feature[i,:,:,jj])
            ax.tick_params(axis="x",bottom="off",top="off",labelbottom="off")
            ax.tick_params(axis="y",left="off",right="off",labelleft="off")
        pyplot.tight_layout()
        #pyplot.show()
        pyplot.savefig("first_conv_feature_"+str(jj)+".png")

        #second convolution features
        fig,axis=pyplot.subplots(ncols=8,nrows=2,figsize=(8,3))
        pyplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.01, hspace=0.001)
        for i,ax in enumerate(axis.ravel()):
            ax.imshow(second_convlayer_feature[i,:,:,jj])
            ax.tick_params(axis="x",bottom="off",top="off",labelbottom="off")
            ax.tick_params(axis="y",left="off",right="off",labelleft="off")
        pyplot.tight_layout()
        #pyplot.show() 
        pyplot.savefig("second_conv_feature_"+str(jj)+".png")

vis_feature(layers)
logger.info("done")
