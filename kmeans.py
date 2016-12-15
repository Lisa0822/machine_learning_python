#! /usr/bin/python
#-*- coding:UTF-8 -*-

######################################
# kmeans: continous data
# Author: Lisa
# Date: 2016-04-12
# email: zhangyanxia2008@126.com 
######################################

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random

def Distance(v1,v2,disType = 'euclidean'):

# calculate euclidean distance
   if disType == 'euclidean':
      return math.sqrt(sum(map(lambda x: math.pow(x[0]-x[1],2), [x for x in zip(v1,v2)])))

# calculate manhattan distance
   elif disType == 'manhattan':
      return sum(map(lambda x: abs(x[0]-x[1]), [x for x in zip(v1,v2)]))

# calculate chebyshev distance
   elif disType == 'chebyshev':
      return max(map(lambda x: abs(x[0]-x[1]), [x for x in zip(v1,v2)]))

   else:
      print 'There is not this type of calculating distance,\n Please choose euclidean or manhattan or chebyshev .'

# init centroids with random samples
def initCentroids(dataSet,k):
   numSample = len(dataSet)
   initIndex = random.sample(range(numSample),k)
   initCentroids = dataSet.iloc[initIndex]
   return initCentroids

# k-means cluster
def kmeans(dataSet, k, disType = 'euclidean', iter_max = 30):
   # init centroids
   Centroids = initCentroids(dataSet,k)
   # define finally iteration count
   iterCnt = 0
   # sum of distance of every sample and Centroids   
   distanceSum = []
   while (iterCnt < 2) or (iterCnt <= iter_max and distanceSum[1] <= distanceSum[-2]) : 
      label = []
      distance = []
      # for each sample
      for i in range(len(dataSet)):
         classDis = []
         # for each centroid
         for j in range(k):
            classDis.append(Distance(dataSet.iloc[i], Centroids.iloc[j], disType))    
         # find the centroid, label and distance who is closest
         label.append(classDis.index(min(classDis)))
         distance.append(min(classDis))
      
      # calculate the sum of distance of all samples   
      distanceSum.append(sum(distance))
      
#      print """iterCnt: {0}
#-------------------------------------------------------------------
#distanceSum:
#   {1}
#-------------------------------------------------------------------
#Centroids::
#   {2} 
#-------------------------------------------------------------------
#   dataSet.groupby(label).count():
#   {3}
#===================================================================
#   """.format(iterCnt, distanceSum, Centroids, dataSet.groupby(label).count())
      iterCnt += 1
      Centroids = dataSet.groupby(label).mean()
      
   summary = {'iterCnt':iterCnt, 'distanceSum':distanceSum, 'Centroids':Centroids, 'label':label} 
   return summary

def plot_kmeans(dataSet,summary):
   label = summary['label']
   label_df = pd.DataFrame({'label':label})
   Centroids = summary['Centroids']
   
   plt.scatter(dataSet.iloc[:,0], dataSet.iloc[:,1], marker='o', s=20, c=label)   
   plt.scatter(Centroids.iloc[:,0], Centroids.iloc[:,1], marker='^', s=100, c=Centroids.index)
   return plt.show()
        
def testIris():
   ## dataSet,label
   iris = pd.read_csv('data/iris.txt',index_col=0)
   dataSet = iris.iloc[:,0:4]
   Species = iris.Species
   
   ## cluster
   k = 3
   #iterCnt, distanceSum, Centroids, label 
   summary = kmeans(dataSet, k, iter_max = 100, disType = 'euclidean' )
   label = summary['label']
   label_pt = pd.DataFrame({'pre':label,'true':Species})
   print """ The matrix of predict and true label is :
-------------------------------------------------------
{0}
-------------------------------------------------------
""".format(pd.pivot_table(label_pt, index='pre', columns='true', aggfunc=np.size))
   plot_kmeans(dataSet,summary)

testIris()
