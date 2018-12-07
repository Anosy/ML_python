import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


def loadDataSet(filename):
    """
    加载数据
    :param filename: 文件名
    :return: xArr-x坐标和b偏差， yArr-y坐标
    """
    numFeat = len(open(filename).readline().split('\t')) - 1
    xArr = []
    yArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


def plotDataSet():
    """
    绘制样本点
    :return:  无
    """
    xArr, yArr = loadDataSet('ex0.txt')
    n = len(xArr)
    xcord = []  # x样本点
    ycord = []  # y样本点
    for i in range(n):
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=0.5)
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.show()


def standRegression(xArr, yArr):
    """
    计算回归系数 ws
    :param xArr: x数据集
    :param yArr: y数据集
    :return: ws回归系数
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:  # 如果xTx的行列式结果为零，说明xTx矩阵不可逆
        print('矩阵为奇异矩阵，不可逆')
    ws = xTx.I * (xMat.T * yMat)  # 利用公式求得ws，mat.I表示矩阵求逆
    return ws


def plotRegression():
    """
    绘制回归线
    :return: 无
    """
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegression(xArr, yArr)
    xMat = np.mat(xArr)  # 将列表转化为矩阵形式
    yMat = np.mat(yArr)  # 将列表转化为矩阵形式
    xCopy = xMat.copy()
    xCopy.sort(axis=0)  # 按列进行排序
    yHat = xCopy * ws  # y = x.T * w ,其中已经包含了偏置b
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:, 1], yHat, c='red')
    ax.scatter(xMat[:, 1].flatten().getA()[0, :], yMat.flatten().getA()[0, :], s=20, c='blue', alpha=.5)
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.show()


def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    使用局部线性加权回归来计算ws权值
    :param testPoint: 测试样本点
    :param xArr: x数据集
    :param yArr: y数据集
    :param k: 高斯核的k,自定义参数
    :return: ws权值
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTwx = xMat.T * (weights * xMat)
    if np.linalg.det(xTwx) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = xTwx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def plotlwlrRegression():
    """
    函数说明:绘制多条局部加权回归曲线
    Parameters:
        无
    Returns:
        无
    Website:http://www.cuijiahua.com/
    Modify:
        2017-11-15
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 系统默认字体
    xArr, yArr = loadDataSet('ex0.txt')  # 加载数据集
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)  # 根据局部加权线性回归计算yHat
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)  # 根据局部加权线性回归计算yHat
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)  # 根据局部加权线性回归计算yHat
    xMat = np.mat(xArr)  # 创建xMat矩阵
    yMat = np.mat(yArr)  # 创建yMat矩阵
    xCopy = xMat.copy()
    xCopy.sort(axis=0)
    srtInd = xMat[:, 1].argsort(0)  # 从小到大排序，返回索引值
    xSort = xMat[srtInd][:, 0, :]  # 将xMat按升序排列.xMat[srtInd]返回的是三维的向量（200，1，2）
    fig, axs = plt.subplots(nrows=3, ncols=1,figsize=(10, 8))
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c='red')  # 绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c='red')  # 绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c='red')  # 绘制回归曲线
    axs[0].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0', FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01', FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003', FontProperties=font)
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def abalone():
    abX, abY = loadDataSet('abalone.txt')
    print(abX)
    # print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
    # for i in range(10):
    #     yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    #     yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    #     yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    #     print('k=0.1时,误差大小为:', rssError(abY[0:99], yHat01.T))
    #     print('k=1  时,误差大小为:', rssError(abY[0:99], yHat1.T))
    #     print('k=10 时,误差大小为:', rssError(abY[0:99], yHat10.T))
    #     print('')
    # print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
    # yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    # yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    # yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    # print('k=0.1时,误差大小为:', rssError(abY[100:199], yHat01.T))
    # print('k=1  时,误差大小为:', rssError(abY[100:199], yHat1.T))
    # print('k=10 时,误差大小为:', rssError(abY[100:199], yHat10.T))
    # print('')
    # print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
    # print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))
    # ws = standRegression(abX[0:99], abY[0:99])
    # yHat = np.mat(abX[100:199]) * ws
    # print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))


if __name__ == '__main__':
    abalone()
