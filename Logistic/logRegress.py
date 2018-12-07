import matplotlib.pyplot as plt
import numpy as np
import random


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt', 'r')
    for line in fr.readlines():  # 逐行读取数据
        lineArr = line.strip().split()  # 去掉句子头尾的空白，且进行切分放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 添加数据,为了方便计算，将x0设为1
        labelMat.append(int(lineArr[2]))  # 添加标签
    fr.close()
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def plotDataSet():  # 绘制数据图像
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)  # 转化为numpy的array数组,因为Matrix无法进行索引
    n = np.shape(dataMat)[0]  # 数据的个数
    xcord1 = []
    ycord1 = []  # 正样本
    xcord2 = []
    ycord2 = []  # 负样本
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])  # 1为正样本
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])  # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)
    plt.title('DataSet')  # 绘制title
    plt.xlabel('x')
    plt.ylabel('y')  # 绘制label
    plt.show()  # 显示


def gradAscent(dataMatIn, classLabels):  # 梯度下降法，但是每次更新权值时都需要对所有的数据进行访问
    dataMatrix = np.mat(dataMatIn)  # 转换为numpy的mat
    # print(dataMatrix)
    labelMat = np.mat(classLabels).T  # 转换为numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)  # 获取行数m,列数n
    alpha = 0.001  # 移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500  # 最大迭代次数
    weights = np.ones((n, 1))  # 设置权重初始值为1
    for k in range(maxCycles):  # 循环
        h = sigmoid(dataMatrix * weights)  # h=1/(1+e^(-theta*x))  ，MatrixA*MatrixB为矩阵乘法，而arrayA*arrayB为向量对应位置相乘
        error = labelMat - h  # 每一个输入的误差
        weights = weights + alpha * dataMatrix.T * error  # 更新权重
    return weights.getA()  # 将矩阵转化为向量的形式


def stocGradAscent0(dataMatrix, classLabels):  # 普通随机梯度下降法
    dataMatrix = np.array(dataMatrix)  # 将列表给向量化
    m, n = np.shape(dataMatrix)  # 获取行m,列n
    alpha = 0.01  # 学习率（步长）为0.01
    weights = np.ones(n)  # np.ones(3) = [[1,1,1]]
    list_weights0 = []
    list_weights1 = []
    list_weights2 = []
    for each in range(200):
        for i in range(m):  # 迭代次数为输入个数
            h = sigmoid(sum(dataMatrix[i] * weights))  # 计算每行向量和权重对应位置相乘的和
            error = classLabels[i] - h  # 计算误差
            weights = weights + alpha * error * dataMatrix[i]  # 更新权重
            list_weights0.append(weights[0])
            list_weights1.append(weights[1])
            list_weights2.append(weights[2])
    return weights, list_weights0, list_weights1, list_weights2


def stocGradAscent1(dataMatrix, classLabels, numIter=150):  # 改进的随机梯度下降法
    dataMatrix = np.array(dataMatrix)  # 将矩阵向量化
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)  # 初始化
    list_weights0 = []
    list_weights1 = []
    list_weights2 = []
    for j in range(numIter): # 迭代次数
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)。j为迭代次数，i为样本下标
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex] * weights))  # 选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h  # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            list_weights0.append(weights[0])
            list_weights1.append(weights[1])
            list_weights2.append(weights[2])
            del (dataIndex[randIndex])  # 删除已经使用的样本，不再重复使用其来更新系数
    return weights, list_weights0, list_weights1, list_weights2


def plotBestFit(weights):  # 绘制回归线
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []  # 正样本
    xcord2 = []
    ycord2 = []  # 负样本
    for i in range(n):  # 根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])  # 1为正样本
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])  # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]  # x2 = (-W0-W1X1)/W2
    ax.plot(x, y)  # 绘制直线
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def plotX(list_weights0, list_weights1, list_weights2):
    x0 = list_weights0
    x1 = list_weights1
    x2 = list_weights2
    y = np.arange(0, 15000, 1)
    fig = plt.figure()
    ax0 = fig.add_subplot(311)
    ax0.plot(y, x0)
    ax0.set_ylabel('X0')  #  设置子图的坐标名字
    ax1 = fig.add_subplot(312)
    ax1.plot(y, x1)
    ax1.set_ylabel('X1')
    ax2 = fig.add_subplot(313)
    ax2.plot(y, x2)
    ax2.set_ylabel('X2')
    plt.xlim(0, 14000)
    plt.ylim(-30, 10)
    plt.xlabel('N number')
    plt.show()


def classifyVector(inX, weights):  #  分类结果，prob以0.5为分界。大于0.5返回1，小于0.5返回0
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest0(): # 使用改进型梯度下降法
    frTrain = open('horseColicTraining.txt')                              #打开训练集
    frTest = open('horseColicTest.txt')                                   #打开测试集
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))                           # lineArr表示每行的各特征的数值
        trainingSet.append(lineArr)                                      # 训练集的特征值
        trainingLabels.append(float(currLine[-1]))                       # 训练集的标签值
    trainWeights, list_weights0, list_weights1, list_weights2 = stocGradAscent1(trainingSet, trainingLabels, 500)       #使用改进的随即上升梯度训练
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():                                     # 读取每行测试文件
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):  # sigmoid(权值*特征)判别类别
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100                                 #错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)
    return errorRate


def colicTest1():  # 】 使用普通梯度下降法
    frTrain = open('horseColicTraining.txt')                              #打开训练集
    frTest = open('horseColicTest.txt')                                   #打开测试集
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))                           # lineArr表示每行的各特征的数值
        trainingSet.append(lineArr)                                      # 训练集的特征值
        trainingLabels.append(float(currLine[-1]))                       # 训练集的标签值
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)        #使用改进的随即上升梯度训练
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():                                     # 读取每行测试文件
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights[0]))!= int(currLine[-1]):  # sigmoid(权值*特征)判别类别
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100                                 #错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)
    return errorRate



def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest0()
    print('运行%d次的平均误差为%.2f%%' %(numTests, errorSum/float(numTests)))


if __name__ == '__main__':
    multiTest()   # 由于数据量较少，也因为其中包含大量的缺失数据，容易发生欠拟合现象

