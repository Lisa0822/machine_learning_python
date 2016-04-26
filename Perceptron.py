#! /usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import random
import sys
from sklearn import metrics
import matplotlib.pyplot as plt

#class perceptron(object):
#   def __int__(self, eta=0.01, iters=50):
#      self.eta = eta
#      self.iters = iters
#   def train(self, X, y):
#      self.w_ = np.zeros(1+X.shape[1])
#      self.errors = []
#
#   for _ in range(self.iters):
#      errors = 0
#      for xi, target in zip(X,y):
#         update = self.eta * (target - self.predict(xi))

# sigmoid
def predict(x, w0, w1, type = 'sigmoid'):
   if type == 'sigmoid':
      return 1/(1 + np.exp(-net_input(x, w0, w1)))
   elif type == 'sgn':
      return np.where(net_input(x, w0, w1)>0, 1, -1)
   elif type == 'doubleSigmoid':
      return (1-np.exp(-net_input(x, w0, w1)))/(1+np.exp(-net_input(x, w0, w1)))
   elif type == 'tanh':
      return (np.exp(net_input(x,w0,w1)) - np.exp(-net_input(x,w0,w1)))/(np.exp(net_input(x,w0,w1)) + np.exp(-net_input(x,w0,w1)))
   else:
      print "Wrong type, please choose : sigmoid, sgn, doubleSigmoid or tanh !"
   
def net_input(x, w0, w1):
   return np.dot(x, w1) + w0

def model(X, y, eta=0.01, iters=50, type = 'sigmoid'):
   #w0 = np.random.random(X.shape)
   # 特征权重
   w0 = 0
   w1 = np.zeros(X.shape[1])
   errors = []
 
   for i in range(iters):      
      pre = predict(X, w0, w1, type) 
      update = eta*(y-pre) 
      w1 += np.dot(X.T, update)
      w0 = sum(update)
      errors.append(sum(update))
   return w0, w1, errors, pre

def rmse(predictions, targets):
   np.sqrt(((predictions-targets)**2).mean())   

def iris_test():
   # load data and split 
   iris = pd.read_csv("data/iris.txt", index_col = 0)
   trainRate = 0.8
   trainSize = int(iris.shape[0] * trainRate)
   
   trainIndex = random.sample(iris.index, trainSize)
   testIndex = list(set(iris.index)-set(trainIndex))
   trainSet = iris.ix[trainIndex]
   testSet = iris.ix[testIndex]

   # train sigmoid
   X = trainSet.ix[:,0:4]
   y = np.where(trainSet.Species =='setosa', 1, 0)
   w0, w1, errors, pre = model(X, y, eta=0.01, iters=500, type = 'sigmoid')  
   
   Species_pre = predict(testSet.ix[:,0:4], w0, w1, type = 'sigmoid')
   Species_true = np.where(testSet.Species == 'setosa', 1, 0)
   fpr,tpr,thresholds = metrics.roc_curve(Species_true, Species_pre, pos_label=1)
   print """The activation function is sigmoid:
----------------------------------------------------------------------------------- 
w0 = {0}
-----------------------------------------------------------------------------------
w1 = {1}
-----------------------------------------------------------------------------------
AUC = {2}
-----------------------------------------------------------------------------------
testSetPre = \n{3}
-----------------------------------------------------------------------------------
fpr, tpr, thresholds = \n{4}
-----------------------------------------------------------------------------------
""".format(w0, w1, metrics.auc(fpr, tpr), pd.DataFrame({"Species_true":Species_true,"Species_pre":Species_pre}), pd.DataFrame({"fpr":fpr,"tpr":tpr,"thresholds":thresholds}))
   plt.plot(fpr, tpr, linestyle ='--', color='b', marker='o', markersize=8)
   plt.ylim(-0.1,1.1)
   plt.xlim(-0.1,1.1)
   plt.show()

iris_test()


