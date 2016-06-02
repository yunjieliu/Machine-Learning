#! /usr/bin/env python

"""
Front events classification NEON model
"""

#lots of module to import
import numpy
import neon
import data_load  #personalized data load module
import ipdb

"""
logging information and command parsing
Can specify these argument on command line or here
"""
from neon.util.argparser import NeonArgparser

parser = NeonArgparser(__doc__)
#argument that are problem dependent
argss=["--file_name","--train_num_p","--valid_num_p","--test_num_p",
       "--train_num_n","--valid_num_n","--test_num_n","--nclass",
       #data dependent arguments
       "--data_dict","--norm_type"  
      ]

for a in argss:
    parser.add_argument(a)

parser.set_defaults(
       #constant arguments
       #rng_seed=2,
       backend= "cpu",
       dataype="f32",
       progress_bar=True,
       log_thresh=10,        
       #variable arguments
       epochs= 15,
       batch_size=100,
       data_dir="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/DATA/",
       file_name="fronts_all.h5",
       nclass=2,
       data_dict=["Front","NonFront"],
       norm_type=2, #1: global contrast norm, 2:standard norm, 3:l1/l2 norm, scikit learn

       #TODO, make the "nlcass" reading from input data, more general

       #follow 80%  20% rule below
       train_num_p=4000,
       valid_num_p=600,
       test_num_p=1000,
       train_num_n=4000,
       valid_num_n=600,
       test_num_n=1000
)

args = parser.parse_args()

#logging information
import logging
logger =logging.getLogger()
logger.setLevel(args.log_thresh)

"""
data loading and sort training, validating and testing
"""
from neon.data import DataIterator

##here the X_train, X_valid, X_test should be a feature vector e.g (3x28x28 will be 2352)
(X_train, Y_train),(X_valid, Y_valid),(X_test,Y_test)= \
         data_load.happy_loader(args.data_dir,args.file_name,
                                  args.data_dict,
                                  args.train_num_p,args.valid_num_p,args.test_num_p,
                                  args.train_num_n,args.valid_num_n,args.test_num_n,
                                  args.norm_type,normalize=True)

train=DataIterator(X_train,Y_train,nclass=args.nclass,lshape=(3,27,60))
valid=DataIterator(X_valid,Y_valid,nclass=args.nclass,lshape=(3,27,60))
test=DataIterator(X_test,Y_test,nclass=args.nclass,lshape=(3,27,60))

logger.info("load data complete...")
"""
default back end gernate will be CPU, here pass that section
"""

"""
construct the model
"""
#initialize weights
from neon.initializers import Uniform,Constant

init_uni = Uniform(low=-0.1, high=0.1)

#learning rule
from neon.optimizers import GradientDescentMomentum

opt_gdm = GradientDescentMomentum(learning_rate=0.05,
                                  momentum_coef=0.9,
                                  stochastic_round=args.rounding,
                                  wdecay=0.001 )

#model layers and its activation
from neon.layers import Affine, Conv, Pooling
from neon.transforms import Rectlin, Softmax, Logistic, Identity

layers=[]
layers.append(Conv((5, 5, 8), init=init_uni, activation=Rectlin(),batch_norm=False))
layers.append(Pooling((2, 2),strides=2))
layers.append(Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=False))
layers.append(Pooling((2, 2),strides=2))
layers.append(Affine(nout=50, init=init_uni,activation=Rectlin(), batch_norm=False))
layers.append(Affine(nout=2, init=init_uni, activation=Logistic()))

#cost function
from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti, CrossEntropyBinary

cost = GeneralizedCost(costfunc=CrossEntropyBinary())

#final model
from neon.models import Model

mlp = Model(layers=layers)

logger.info("model construction complete...")

"""
model training and classification accurate rate
"""
#model training and results
from neon.callbacks.callbacks import Callbacks

callbacks = Callbacks(mlp, train, args,eval_set=train)

#run the model
mlp.fit(train, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

#classification accuracy
from neon.transforms import Misclassification

t_mis_rate=mlp.eval(train, metric=Misclassification())*100
v_mis_rate=mlp.eval(valid, metric=Misclassification())*100
#test_mis_rate=mlp.eval(test, metric=Misclassification())*100

print ('Train Misclassification error = %.1f%%' %t_mis_rate)
print ('Valid Miscladdifcaiton error = %.1f%%' %v_mis_rate)
#print ('Test Miscladdifcaiton error = %.1f%%' %test_mis_rate)


"""
filters and features learned by the model
"""

