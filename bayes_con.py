#! /usr/bin/python
# -*- coding:UTF-8 -*-

######################################
# bayes: continous data
# Author: Lisa
# Date: 2016-04-11
# Email: zhangyanxia2008@126.com

#######################################

import numpy as np
import pandas as pd
import random
import math

def loadData():
   dataSet = pd.read_csv("/Users/zhyx/Desktop/iris.txt",index_col=0)
   dataSet = dataSet.rename(columns={"Species":"Class"})
   return dataSet

def splitDataSet(dataSet,splitRatio):
   trainSize = int(len(dataSet)*splitRatio)
   train_ix = random.sample(range(1, len(dataSet) + 1), trainSize)
   train_ix.sort()
   trainSet = dataSet.ix[train_ix]
   test_ix = list(set(range(1, len(dataSet) + 1))^set(train_ix))
   test_ix.sort()
   testSet = dataSet.ix[test_ix]
   return trainSet, testSet

def mean(numbers):
   return sum(numbers)/float(len(numbers))

def stdev(numbers):
   avg = mean(numbers)
   variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
   return math.sqrt(variance)

#def distanceEucl(trainSet):
#   centers=trainSet.groupby(trainSet['Species']).mean()
#   for cl in list(set(trainSet['Species'])):
#      trainSet[]

#def summarize(dataSet):
#   summaries =[(mean(attribute), stdev(attribute)) for attribute in zip(*dataSet)]
#   del summaries[-1]
#   return summaries

def summarizeByClass(trainSet):
   meanByClass=trainSet.groupby(trainSet['Class']).mean()
   stdByClass=trainSet.groupby(trainSet['Class']).std()
   summaries={}
   for cl in list(set(trainSet['Class'])):
      summaries[cl]=zip(meanByClass.ix[cl], stdByClass.ix[cl])
   return summaries

def calculateProbability(x,mean,std):
   exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
   return (1/(math.sqrt(2*math.pi)*std))*exponent

def calculateClassProbabilities(summaries,inputVect):
   probabilities = {}
   for cl in summaries.keys():
      for col in range(len(inputVect)):
         probability = calculateProbability(inputVect[col],summaries[cl][col][0],summaries[cl][col][1])
         if col==0:
            probabilities[cl] = probability
         else:
            probabilities[cl] *= probability 
   return probabilities      

def classJudge(probabilities):
   classJudge=[]
   for cl in probabilities.keys():
      if probabilities[cl] == max(probabilities.values()):
         classJudge = cl
   return classJudge

def getTrainSetRes(trainSet):
   #trainSet,testSet = splitDataSet(dataSet,0.8)
   summaries=summarizeByClass(trainSet)   
   classPre = []
   for i in trainSet.index:
      probByClass = calculateClassProbabilities(summaries,trainSet.iloc[:,0:4].ix[i])
      classPre.append( classJudge(probByClass) )
   classData = pd.DataFrame({'classTrue':list(trainSet.Class), 'classPre':classPre }) 
   rightRate = float(len(classData[classData.classTrue==classData.classPre]))/len(classData) 
   wrongRate = 1.0 - rightRate 
   return summaries,classData,rightRate,wrongRate 

def getPrediction(summaries,testSet):
   #summaries=summarizeByClass(trainSet)
   testData=testSet.iloc[:,0:4]
   classPre = []
   for i in testData.index:
      probByClass = calculateClassProbabilities(summaries,testData.ix[i])
      classPre.append( classJudge(probByClass) )
   testSet['classPre'] = classPre
   rightRate = float(len(testSet[testSet.Class==testSet.classPre]))/len(testSet)
   return testSet, rightRate

def main():
   filename = '/Users/zhyx/Desktop/iris.txt'
   splitRatio = 0.8
   dataSet = loadData()
   trainSet,testSet=splitDataSet(dataSet,splitRatio)
   print('Split %d rows into train=%d and test=%d rows') % (len(dataSet), len(trainSet), len(testSet))
   
   print('trainSet: {0}').format(trainSet.groupby(trainSet.Class).count())
   summaries=summarizeByClass(trainSet)
   predictions, accuracy=getPrediction(summaries, testSet)
   print ('Accuracy: {0}').format(accuracy)

main()
