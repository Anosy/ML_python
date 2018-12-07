import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



def loadSimpData():  # 加载样本数据
    dataMat = np.matrix([
        [1., 2.1],
        [1.5, 1.6],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def showDataSet(dataMat, classLabels): # 显示数据分布
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(data_plus_np.T[0],data_plus_np.T[1])
    plt.scatter(data_minus_np.T[0],data_minus_np.T[1])
    plt.show()


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):  # 单层决策树生成函数
    """
    :param dataMatrix: 数据矩阵
    :param dimen:  第几个特征
    :param threshVal:  特征阈值
    :param threshIneq: 符号，lt：小于(less than)，gt：大于(great than)
    """
    retArray = np.ones((np.shape(dataMatrix)[0],1))   # 初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0 # 如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0  # 如果大于阈值,则赋值为-1
    return retArray


def buildStump(dataArr,classLabels,D):
    """
    目的是为了遍历所有的数据，找到最佳的分类特征，以及其特征的值，以及特征的符号，构建最佳的单层决策树。
    :param dataArr: 数据矩阵
    :param classLabels: 数据标签
    :param D:   样本权重
    :return:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = float('inf')  # 将最小误差初始化为无穷大
    for i in range(n):  # 遍历所有特征
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()  # 找到特征中最小的值和最大值
        stepSize = (rangeMax - rangeMin) / numSteps  # 计算步长
        for j in range(-1, int(numSteps) + 1):
            for unequal in ['lt', 'gt']:  # 大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)  # 计算阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, unequal)  # 计算分类结果
                errArr = np.mat(np.ones((m, 1)))  # 初始化误差矩阵
                errArr[predictedVals == labelMat] = 0  # 分类正确的,赋值为0
                weightedError = D.T * errArr  # 计算误差
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                # i, threshVal, unequal, weightedError))
                if weightedError < minError:  # 找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['uneq'] = unequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):  # Adaboost算法
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)   # 初始化权重
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # 构建单层决策树
        # print("D:", D.T)
        alpha = float(0.5 * np.log((1-error) / max(error, 1e-16))) # 由于error不能为零，因此用1e-16近似
        bestStump['alpha'] = alpha  # 存储弱学习算法权重
        weakClassArr.append(bestStump)  # 存储单层决策树
        # print("classEst: ", classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)  # 计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()  # 根据样本权重公式，更新样本权重
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))  # 计算误差
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate)
        if errorRate == 0.0: break  # 误差为0，退出循环
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):  # 测试算法效果
    """
    AdaBoost分类函数
    :param daToClass:  待分类样例
    :param classifierArr:  训练好的分类器
    :return:  分类的结果
    """
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):  # 遍历所有分类器，进行分类
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['uneq'])  # 每一个弱分类器进行分类得到结果
        #print('classEst:',classEst)
        aggClassEst += classifierArr[i]['alpha'] * classEst    # 将每个弱分类器的结果乘上a的值如何加权
        #print(aggClassEst)
    return np.sign(aggClassEst)  # 对结果去符号函数，将大于0的判决为+1类，小于0的判决为-1类。


def loadDataSet(filename):
    # 第一行的作用自动检测特征的数量
    numFeat = len(open(filename).readline().split('\t'))    # readline每次只读一行，readlines全部都读，但是将结果的每一行都保存在列表中
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def plotROC(predStrengths, classLabels):
    """
    绘制ROC曲线
    :param predStrengths:  分类器的预测强度
    :param classLabels: 类别
    :return: 无
    """
    cur = (1.0, 1.0)  # 绘制光标的位置
    ySum = 0.0        # 用于计算AUC的值
    numPosClas = sum(np.array(classLabels)==1.0)    # 统计正类的数量
    yStep = 1/float(numPosClas)                      # y轴步长,1/正类的数量
    xStep = 1/float(len(classLabels)-numPosClas)    # x轴步长，1/负类的数量

    sortedIndicies = predStrengths.argsort()        # 预测强度排序,np.argsort()根据元素的从小到大进行排列，得到对应index索引的位置。
    fig = plt.figure()
    fig.clf()  #  清空figure
    ax = fig.add_subplot(111)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']       # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False          # 解决保存图像是负号'-'显示为方块的问题
    for index in sortedIndicies.tolist()[0]:  # 由于索引是按照从小到大的排序，预测如果出现等于1表示预测错误，应该降低y的坐标值
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]  # 高度累加
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')  # 绘制ROC
        cur = (cur[0] - delX, cur[1] - delY)  # 更新绘制光标的位置
    ax.plot([0, 1], [0, 1], 'b--')
    plt.title(u'AdaBoost马疝病检测系统的ROC曲线')
    plt.xlabel(u'假阳率')
    plt.ylabel(u'真阳率')
    ax.axis([0, 1, 0, 1])
    print('AUC面积为:', ySum * xStep)  # 计算AUC
    plt.show()

if __name__ == '__main__':
    dataArr, classLabels = loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
    plotROC(aggClassEst.T, classLabels)
