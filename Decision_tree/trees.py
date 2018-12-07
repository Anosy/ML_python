from math import log
import operator


def calcShannonEnt(dataSet):  # 计算香农熵
    numEntries = len(dataSet)  # 计算数据集长度
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 获取标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0  # 字典的value如果为空时，不能直接对其进行‘+=’符号
        labelCounts[currentLabel] += 1    # 以字典的形式进行存储
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  # 计算公式
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():   # 创建数据集
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):  # 对数据集进行划分。参数分别是待划分的数据集，划分的数据集特征，需要返回的特征的值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])  # 注意extend,append的区别
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):   # 通过划分后的熵的计算，选择最好的数据集划分的特征
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算原始的信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]   # 以dataSet中的第i个特征，构建list
        uniqueVals = set(featList)  # 将列表转化为集合set，目的是取到唯一元素
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)   # 对数据集的第i个特征，以不同的取值进行划分
            prob = len(subDataSet)/float(len(dataSet))   # 计算划分后的比例
            newEntropy += prob * calcShannonEnt(subDataSet)  # 以上三句，计算出属性a上取值为a1的信息熵
        infoGain = baseEntropy - newEntropy  # 计算信息增益 公式见机器学习P75页
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):   # 如果使用完了所有的特征，但是还是不能将数据集划分为唯一的分组，则使用最多的分类来作为结果
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 用列表推导式来创建列表，元素都是为标签
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同的话，那么停止划分
        return classList[0]  # 返回类别
    if len(dataSet[0]) == 1:   # 如果元素的个数只剩为1，即遍历了所有的特征
        return majorityCnt(classList)  # 返回出现次数最多的
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最好的划分特征
    bestFeatLabel = labels[bestFeat]  # 最好的划分特征名
    myTree = {bestFeatLabel: {}}
    subLabels = labels[:]
    del(subLabels[bestFeat])  # 删除当前的最好特征，以便后面划分后选择最好特征
    featValues = [example[bestFeat] for example in dataSet]  # 最好划分特征的特征值构成list
    uniqueVals = set(featValues)    # 消除重复，确定特征值
    for value in uniqueVals:  # 以特征值进行划分，构建决策树
        # subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree,featLabels,testVec):   # 测试训练的结果
    firstStr = inputTree.keys()
    for each in firstStr:  # 由于获取的firstStr，类型为字典的键类型，使用此方法来进行转换
        Str = each
    secondDict = inputTree[Str]
    featIndex = featLabels.index(Str)  # firstStr的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):  # 存储决策树
    import pickle
    fw = open(filename, 'wb')    # pickle用二进制来写入文件
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):   # 读取决策树
    import pickle
    fr = open(filename, 'rb')   # pickle用二进制来读取文件
    return pickle.load(fr)



