#######################################
#! /usr/bin/python
#-*- coding:UTF-8 -*-

# bayes: 0-1 classify
# Author: Lisa
# Date: 2016-04-08
# Email: zhangyanxia2008@126.com

#######################################

from numpy import *
##词表到向量的转化函数
#加载数据
def loadDataSet():
	postingList=[['my','dog','has','flea','problems','help','please'],
		['maybe','not','take','him','to','dog','park','stupid'],
		['my','dalmation','is','so','cute','I','love','him'],
		['stop','posting','stupid','worthless','garbage'],
		['mr','licks','ate','my','steak','how','to','stop','him'],
		['quit','buying','worthless','dog','food','stupid']]
	classVec=[0,1,0,1,0,1]    #1代表侮辱性文字，0代表正常言论
	return postingList,classVec

#将加载的数据转化为list
def createVocabList(dataSet):
	#创建一个空集
	vocabSet = set([])
	for document in dataSet:
		#创建两个集合的并集
		vocabSet = vocabSet | set(document)
	return list(vocabSet)


##词集模型：输入参数inputSet是否在输入文档vocabList中出现，出现返回1，未出现，返回0
def setOfWords2Vec(vocabList, inputSet):
	#创建一个其中所含元素都是0的向量
	returnVec=[0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print "the word: %s is not in my Vocabulary" % word
	return returnVec

#词袋模型：输入参数inputSet是否在输入文档vocabList中出现的次数
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#朴素贝叶斯分类器训练函数
#trainMatrix:文档矩阵
#trainCategory：每篇文档类别构成的向量
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #训练文档的数目
    numWords = len(trainMatrix[0])	#训练文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)	#计算文档属于侮辱性文档的概率
    #❶ （以下两行）初始化概率 
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    #一旦某个词在某一文档中出现，则改词对应的个数就＋1，在所有文档中，该文档的总词数＋1
    #判断该文档可能是因为哪个词导致称为侮辱性文档／正常文档
    for i in range(numTrainDocs): #对每篇训练文档
        if trainCategory[i] == 1: #如果是侮辱性文档
            #❷（以下两行）向量相加 
            p1Num += trainMatrix[i]	#增加该词条的计数值：侮辱性文档中的每个词在词库中出现的次数
            p1Denom += sum(trainMatrix[i]) #增加所有词条的计数值:侮辱性文档中的词在词库中出现的次数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    #❸ 对每个元素做除法
    p1Vect = log(p1Num/p1Denom) #change to log() #每个词是侮辱词的概率
    p0Vect = log(p0Num/p0Denom) #change to log() #每个词是正常词的概率
    return p0Vect,p1Vect,pAbusive #返回每个类别的条件概率

#vec2Classify:要分类的向量
#p0Vec, p1Vec, pClass1:trainNB0()得到的三个概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #❶ 元素相乘
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb) 


def myTestBayes(testEntry):
     postingList,classVec=loadDataSet()
     myVocabList=createVocabList(postingList)
     trainMat=[]
     for postInDoc in postingList:
             trainMat.append(setOfWords2Vec(myVocabList,postInDoc))
     p0V,p1V,pAb = trainNB0(array(trainMat),array(classVec))
     thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
     print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        #❶ （以下七行）导入并解析文本文件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50); testSet=[]
    #❷（以下四行）随机构建训练集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    #❸（以下四行）对测试集分类
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
           errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)  
