#######################################
#!/usr/bin/python
#-*- coding:UTF-8 -*-

# KNN: classify hand-writing data
# Author: Lisa
# Date: 2016-04-12
# Email: zhangyanxia2008@126.com

#######################################

import numpy as np
import pandas as pd
import os
import math

def img2Vect(imgData):
   imgDataList = list(imgData.ix[:,0])
   dataVectStr = []
   for ls in imgDataList:
      dataVectStr += list(ls)   
   dataVect = map(lambda x: int(x), [x for x in dataVectStr])
   return dataVect

def loadData(dataSetDir):
   #dataSetDir='data/knnData/trainingDigits'
   dataList=os.listdir(dataSetDir)
   numSamples = len(dataList)
   label = []
   dataSet = pd.DataFrame()
   for i in range(numSamples):
      imgData = pd.read_csv(dataSetDir + '/' + dataList[i],header=None)
      dataSet[dataList[i]] = img2Vect(imgData)
      label.append(int(dataList[i].split('_',1)[0]))
   dataSet = dataSet.T
   dataSet['label'] = label
   return dataSet

def EuclideanDis(v1,v2):
   EuclDis=math.sqrt(sum(map(lambda x: math.pow(x[0]-x[1],2), [x for x in zip(v1,v2)]))) 
   return EuclDis

def classPredict(trainSet,inputVec,K):
   distance = []
   for i in range(len(trainSet)):
      distance.append( EuclideanDis(trainSet.iloc[i,0:1023], inputVec))
   EuclDis = pd.DataFrame({'distance':distance, 'label':trainSet.label},index = trainSet.index)
   EuclDis = EuclDis.sort('distance')
   EuclDisK = EuclDis.iloc[0:K,]
   classCnt = EuclDisK.groupby(EuclDisK.label).count()
   labelPre=classCnt[classCnt.distance == max(classCnt.distance)].index[0]
   return labelPre

def knnClassify(trainSet,testSet,K):
   labelPre = []
   for i in range(len(testSet)):
      labelPre.append(classPredict(trainSet, testSet.iloc[i,0:1024],K))
   testSet['labelPre'] = labelPre
   accuracy = len(testSet[testSet.label==testSet.labelPre])/float(len(testSet))
   return testSet, accuracy

def testHandWriting():
   trainSet = loadData('data/knnData/trainingDigits')
   testSet = loadData('data/knnData/testDigits')
   K = 30
   classPreData,accuracy = knnClassify(trainSet,testSet,K)
   print 'The knn classify accuracy of hand-writing Data is: %.2f%% .' % (accuracy * 100)
