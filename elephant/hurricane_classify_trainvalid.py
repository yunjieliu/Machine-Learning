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
       epochs= 2,
       batch_size=100,
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
       out_dir="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/RESULTS/hurricane/",
       save_path="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/RESULTS/hurricane/hurricane_classify_train.pkl",
       serialize= 2,
       logfile="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/RESULTS/hurricane/hurricane_classify_train."+c_time+".log",
       output_file="/global/project/projectdirs/nervana/yunjie/climate_neon1.0run/conv/RESULTS/hurricane/hurricane_classify_train."+c_time+".h5",
) 

args = parser.parse_args()

"""
data loading and sort training, validating and testing
"""

##here the X_train, X_valid, X_test should be a feature vector e.g (3x28x28 will be 2352)
(X_train, Y_train),(X_valid, Y_valid),(X_test,Y_test)= \
         data_load.happy_loader(args.rng_seed,args.data_dir,args.file_name,
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
layers.append(Conv((5, 5, 16), init=init_uni, activation=Rectlin(),batch_norm=False))
layers.append(Pooling((2, 2),strides=2))
layers.append(Conv((5, 5, 32), init=init_uni,activation=Rectlin(), batch_norm=False))
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
#model training and results

callbacks = Callbacks(mlp,train, args, eval_set=valid,metric=Misclassification())

#add lost and metric call backs facilitate more diagnostic

callbacks.add_callback(MetricCallback(mlp,eval_set=train,metric=Misclassification(),epoch_freq=args.evaluation_freq))
callbacks.add_callback(MetricCallback(mlp,eval_set=valid,metric=Misclassification(),epoch_freq=args.evaluation_freq))
#run the model

mlp.fit(train, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

#final classification accuracy

t_mis_rate=mlp.eval(train, metric=Misclassification())*100
v_mis_rate=mlp.eval(valid, metric=Misclassification())*100
#test_mis_rate=mlp.eval(test, metric=Misclassification())*100

print ('Train Misclassification error = %.1f%%' %t_mis_rate)
print ('Valid Miscladdifcaiton error = %.1f%%' %v_mis_rate)
#print ('Test Miscladdifcaiton error = %.1f%%' %test_mis_rate)

