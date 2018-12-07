'''
使用sklearn.svm.SVC  进行手写数字的分类
SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False,
        tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’,
        random_state=None)
参数：
    C：惩罚项，float类型，可选参数，默认为1.0，C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，也就是对测试数据的分类准确率降低。
    kernel：核函数类型，str类型，默认为’rbf’可选参数为： ’linear’：线性核函数‘poly’：多项式核函数‘rbf’：径像核函数/高斯核‘sigmod’：sigmod核函数‘precomputed’：核矩阵
    degree：多项式核函数的阶数，int类型，可选参数，默认为3。只对多项式核函数有用，如果其他核函数将忽略该问题
    gamma：核函数系数，float类型，可选参数，默认为auto。只对’rbf’ ,’poly’ ,’sigmod’有效。如果gamma为auto，代表其值为样本特征数的倒数，即1/n_features。
    max_iter：最大迭代次数，int类型，默认为-1，表示不限制。
    class_weight：类别权重，dict类型或str类型，可选参数，默认为None。给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C。
        如果给定参数’balance’，则使用y的值自动调整与输入数据中的类频率成反比的权重。
'''

import numpy as np
from sklearn.svm import SVC
from os import listdir
import operator
import matplotlib.pyplot as plt


def img2Vector(filename):  # 将32*32的图片给转化为1024的向量
    returnVect = np.zeros((1,1024))
    fr = open(filename,'r')
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def read_train():  # 读取训练集
    hwLabels = []
    trainFileList = listdir('trainingDigits')
    m = len(trainFileList)
    trainMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainMat[i, :] = img2Vector('trainingDigits/%s' % (fileNameStr))
    return trainMat, hwLabels


def read_test():  # 读取测试集
    hwLabels = []
    testFileList = listdir('testDigits')
    m = len(testFileList)
    testMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        testMat[i, :] = img2Vector('testDigits/%s' % (fileNameStr))
    return testMat, hwLabels


def handwriting():
    trainMat, trainLabels = read_train()
    testMat, testLabels = read_test()
    clf = SVC(C=500, kernel='rbf', gamma=0.01)
    clf.fit(trainMat, trainLabels)
    errorCount = 0.0
    mTest = testMat.shape[0]
    for i in range(mTest):
        vectorUnderTest = np.array(testMat[i, :]).reshape(1, -1)  # 将其转化为array格式，然后用reshape将其转为2维，如果不用reshape仅仅为一个列表的格式
        Result = clf.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (Result, testLabels[i]))
        if Result != testLabels[i]:
            errorCount += 1.0
    score = clf.score(testMat, testLabels)
    print(score)
    print("总共错了%d个数据\n准确率为%f%%" % (errorCount, ((1 - errorCount / mTest) * 100)))


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[-1]))
    return dataMat, labelMat


def showDataSet(dataMat, labelMat):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(data_plus_np[:,0],data_plus_np[:,1])
    plt.scatter(data_minus_np[:,0], data_minus_np[:,1])
    plt.show()

def linearSVC(dataMat,labelMat):
    clf = SVC(kernel='linear',C=1)
    clf.fit(dataMat, labelMat)



if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    showDataSet(dataMat, labelMat)






