from numpy import *
import feedparser # 通用供稿解析器,处理RSS 0.9x，RSS 1.0，RSS 2.0等


def loadDataSet():  # 加载数据
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec  # postingList--实验样本切分的词条， classVec--类别标签向量


def createVocabList(dataSet):   # 创建的词汇表，其中的字符不不具有重复性
    vocabSet = set([])  # 创建一个空集,集合形式
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 创建两个集合的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):  # 获得词条向量
    returnVec = [0] * len(vocabList)        # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1     # index函数在字符串里找到字符第一次出现的位置  词集模型
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def bagOfWords2Vec(vocabList, inputSet):  # 获得词条向量
    returnVec = [0] * len(vocabList)        # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] =+ 1  # index函数在字符串里找到字符第一次出现的位置,如果多次出现，这可以添加多次
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):  # 贝叶斯分类器训练函数
    numTrainDocs = len(trainMatrix)  # 计算训练文档的数量，即setOfWords2Vec返回的returnVec构成的矩阵
    numWords = len(trainMatrix[0])   # 计算每一篇文档的词条数，即又多少个不重复的词语
    pAbusive = sum(trainCategory) / float(numTrainDocs)   # 文档属于侮辱类的概率，sum(trainCategory)表示计算侮辱类的个数
    p0Num = ones(numWords) ; p1Num = ones(numWords) # 构建全为1的，长度为numWords的一维向量,防止其中一个概率为0时，导致所有概率乘为0
    p0Denom = 2.0 ;  p1Denom = 2.0  # 分母初始化
    for i in range(numTrainDocs):
        if trainCategory[i] == 1 :    # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]    # p1num 获取所有侮辱类文档中，每个对于词条出现的数量
            p1Denom += sum(trainMatrix[i])  # 得到所有侮辱类文档，侮辱词条的个数
        else:                          # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # print('p1Num %s' % p1Num)
    # print('p1Denom %s' % p1Denom)
    p1Vect = log(p1Num / p1Denom)   # 计算P(w0|1),P(w1|1),P(w2|1)···  向量形式
    p0Vect = log(p0Num / p0Denom)    # 计算P(w0|0),P(w1|0),P(w2|0)···  向量形式
    return p0Vect, p1Vect, pAbusive  # pAbusive文档属于侮辱类的概率


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):  # vec2Classify--要分类的向量，其他三个为trianNB0对应的输出
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)   #  将要分类的输入向量的对应位置乘上对应位置的概率，然后相加，最后加上类的概率
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)  # logA * B = logA + logB
    if p1 > p0:  # 如果p1>p0的话，那么判断该类别为1类，反之亦然。
        return 1
    else:
        return 0


def testingNB():  # 测试贝叶斯分类器的效果
    list0Posts, listClasses = loadDataSet()  # 加载训练数据集
    myVocabList = createVocabList(list0Posts) # 创建个人词汇表
    trainMat = []
    for postinDoc in list0Posts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 向量化词汇表
    # print(array(trainMat))
    # print(array(listClasses))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))  # 训练贝叶斯模型，输出p0V，p1V向量和pAb概率
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList ,testEntry))  # 向量化该输入，得到该输入在词汇表中对于位置为1的向量
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果

    testEntry = ['stupid', 'garbage']  # 测试样本2
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果


def textParse(bigString):  # 构建字符列表，将字符变为小写，且去掉长度小于2的
    import re
    listOfTokens = re.split('\W+', bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]


def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):     # 导入ham和spam，并且将其解析为词列表
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)    # 文档列表，不同文章内容用list分割
        fullText.extend(wordList)   # 全文列表，将全部文章中的字，保存在一个列表中
        classList.append(1)   # 类别列表，用'1'代表spam,用'0'代表ham
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # print(classList)
    # print(fullText)
    vocabList = createVocabList(docList)   # 构建词汇表，表中的内容不重复
    # print(vocabList)
    trainingSet = range(50); testSet = []  # 本例中，有50封邮件
    trainingSet = list(trainingSet)
    for i in range(10):   # 随机选取10封邮件，将其加入到测试集中，且在50封中去除
        randIndex = int(random.uniform(0, len(trainingSet)))
        # print(randIndex)
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # 获取词条向量
        trainClasses.append(classList[docIndex])   # 取对于的类别
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))  # 训练模型，获取p0V, p1V, pSpam
    errorCount = 0
    for docIndex in testSet:
        wordVector  = setOfWords2Vec(vocabList, docList[docIndex])  # 获取词条向量
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 验证集验证，如果出现错误，那么error+1
            errorCount += 1
            print('分类出现错误的测试集 %s' % docList[docIndex])
    print('错误率为： %.2f%%' % (float(errorCount)/len(testSet)*100))


if __name__ =='__main__':
    spamTest()
