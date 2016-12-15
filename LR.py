#! /usr/bin/python
# -*- coding:utf-8 -*-

######################################
# LR: logistic regression
# Author: Lisa
# Date: 2016-04-27
# email: zhangyanxia2008@126.com
##########################################

import numpy as np
import pandas as pd
import random 
import theano
import theano.tensor as T
from sklearn import metrics 
import matplotlib.pyplot as plt
import os, sys

# load data
iris = pd.read_csv("data/iris.txt", index_col=0)

## define inputs
x = T.matrix("x")
y = T.vector("y")
alpha = T.dscalar("alpha")

# initial weights and bias
w = theano.shared(np.zeros(iris.shape[1]-1), name='w')
b = theano.shared(0.,name='b')
print """initial weights and bias: 
   w = {0}
   b = {1}""".format(w.get_value(),b.get_value())
print "--------------------------------------------------------"

#calculate probability and cost 
p_1 = 1/(1+T.exp(-T.dot(x,w)-b))
xcent = -y*T.log(p_1)-(1-y)*T.log(1-p_1)
cost = xcent.mean() + (alpha**2) * (w**2).sum()

# w, b   derivation function
gw,gb = T.grad(cost,[w,b])
# theano.pp()

# construct LR
train = theano.function(
   inputs = [ x, y, alpha],
   outputs = [ p_1, xcent, cost ],
   updates = ((w, w-alpha*gw), (b, b-alpha*gb))
)
predict = theano.function(
   inputs = [ x ],
   outputs = [ p_1 ]
)

#iters = 1000
if len(sys.argv) < 3 :
   #print 'len(sys.argv) = {0} : learning_rate={1}, iters={2}'.format(len(sys.argv), sys.argv[1], sys.argv[2])
   print 'please input learning rate and the max iters:'
else:
   print """len(sys.argv) = {0} :
   learning_rate={1}
   iters={2}""".format(len(sys.argv), sys.argv[1], sys.argv[2])
   for i in range(int(sys.argv[2])):
      pre, err, cs = train(iris.ix[:,0:4], np.where(iris.Species=='setosa',1,0), alpha=float(sys.argv[1]))

   print '--------------------------------------------------------'
#predict(iris.ix[:,0:4])
   print """weights and bias after training:
   w = {0}
   b = {1}""".format(w.get_value(),b.get_value())

   fpr, tpr , thresholds = metrics.roc_curve(np.where(iris.Species=='setosa',1,0), predict(iris.ix[:,0:4])[0])

   print '--------------------------------------------------------'
   print "AUC = {0}".format(metrics.auc(fpr,tpr))
   plt.plot(fpr, tpr, linestyle ='--', color = 'b', marker = 'o', markersize=10)
   plt.ylim(-0.1,1.1)
   plt.xlim(-0.1,1.1)
   plt.show()


