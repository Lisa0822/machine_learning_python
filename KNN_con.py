#######################################
#! /usr/bin/python
#-*- coding:UTF-8 -*-

# KNN: continous data 
# Author: Lisa
#Date: 2016-04-11
# Email: zhangyanxia2008@126.com

#######################################

import pandas as pd
import random
import math

def loadData():
   dataSet = pd.read_csv('/Users/zhyx/Desktop/iris.txt',index_col=0)
   dataSet = dataSet.rename(columns={"Species":"Class"})
   return dataSet

def splitDataSet(dataSet,splitRatio):
   trainSize = int(len(dataSet)*splitRatio)
   trainInd = random.sample(range(1,len(dataSet)+1),trainSize) 
   trainSet = dataSet.ix[trainInd]
   testInd = list(set(range(len(dataSet)))^set(trainInd))
   testSet = dataSet.ix[testInd]
   return trainSet, testSet

def EucludeanDis(v1,v2):
   minusPowList = list(map(lambda x: math.pow(x[0]-x[1],2),zip(v1,v2)))
   EucDis = math.sqrt(sum(minusPowList))
   return EucDis

def Predict(trainSet,inputVec,K):
   EucDis = []
   for i in trainSet.index:
      EucDis.append(EucludeanDis(inputVec,trainSet.iloc[:,0:4].ix[i])) 
   EucDis = pd.DataFrame({'EucDis':EucDis,'Class':trainSet.Class},index = trainSet.index)
   EucDis = EucDis.sort(['EucDis'])
   EucDisK = EucDis.iloc[0:K]
   classCnt = EucDisK.groupby(EucDisK.Class).count()
   classPre = classCnt[classCnt.EucDis==max(classCnt.EucDis)].index[0]   
   return classPre

def knnClassify(trainSet,testSet,K):
   classPre = []
   for i in testSet.index:
      classPre.append(Predict(trainSet,testSet.iloc[:,0:4].ix[i],K))
   testSet['classPre']=classPre
   accuracy = float(len(testSet[testSet.Class==testSet.classPre]))/len(testSet)
   return testSet,accuracy
   
def main():
   dataSet = loadData()
   trainSet,testSet = splitDataSet(dataSet,0.8)
   inputVect =  testSet.iloc[1,0:4]
   outputClass = Predict(trainSet,inputVect,30)
   print """Your inputVect is 
------------------------
{0}
------------------------ 
and classify is '{1}'.
""".format(inputVect, outputClass)

main()  
   





