import matplotlib.pyplot as plt
import numpy as np
import random







def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[-1]))
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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_plus_np[:, 0], data_plus_np[:, 1])
    ax.scatter(data_minus_np[:, 0], data_minus_np[:, 1])
    plt.show()


def selectJrand(i, m):
    j = i  # 选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)  # 转换为numpy的mat存储
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))  # 初始化alphas值，设为0
    iter_num = 0  # 初始化迭代次数
    while (iter_num < maxIter):  # 最多迭代matIter次
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b  # 步骤1，计算误差
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or (
                (labelMat[i] * Ei > toler) and (alphas[i] > 0)):  # 优化alpha，更设定一定的容错率。
                j = selectJrand(i, m)  # 随机选择另一个与alpha_i成对优化的alpha_j
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b  # 步骤1：计算误差Ej
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()  # 保存更新前的alpha值，使用深拷贝。直接赋值-原列表变化，则现对象会发生改变;深拷贝-原列表发生变化，则现对象不发生改变
                alphaJold = alphas[j].copy()  # 保存更新前的alpha值，使用深拷贝
                if (labelMat[i] != labelMat[j]):  # 计算上下界L和H
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print("L==H"); continue     # 如果上界=下界，那么则将跳出循环，进行下一次循环
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j,:] * dataMatrix[j,:].T    # 步骤3：计算eta
                if eta >= 0: print("eta>=0"); continue  # eta表示学习速率，但是规定为负值（因为公式刚好相反）。如果不符合，跳出循环，进行下一次循环
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta   # 步骤4：更新alpha_j
                alphas[j] = clipAlpha(alphas[j], H, L)  # 步骤5：修剪alpha_j
                if (abs(alphas[j] - alphaJold) < 0.00001): print("alpha_j变化太小"); continue   # 如果aj更新太小，跳出循环，进行下一次循环
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])   # 步骤6：更新alpha_i，可以看出，ai，aj改变的大小一样，但是改变的方向不同
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j,:].T  # 步骤7：更新b_1和b_2
                if (0 < alphas[i]) and (C > alphas[i]):   # 步骤8：根据b_1和b_2更新b
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1  # 统计优化次数
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, alphaPairsChanged))  # 打印统计信息
        if (alphaPairsChanged == 0):  # 更新迭代次数
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数: %d" % iter_num)
    return b, alphas


def showClassifer(dataMat, w, b):
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)  # 负样本散点图
    # 绘制直线
    x1 = max(dataMat)[0]    # 选取图形最远的x
    x2 = min(dataMat)[0]    # 选取图形最近的x
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2  # 由于直线为(w1,w2)*(x1,x2).T +b = 0,即w1x1+w2x2+b=0。从而x2 = (-b-w1*x1)/w2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    for i, alpha in enumerate(alphas):    # enumerate()输出(0, seq[0]), (1, seq[1]), (2, seq[2])...
        if abs(alpha) > 0:  # 如果alphas为0的点，表示对SVM的构成无作用，也就是说其为非支持向量机，相反alphas>0的话，说明对应的x点为支持向量
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')  # s属性表示大小
            # b1 = y + (a1/a2) * x
            # plt.plot([0,x],[b1,y])
    plt.show()


def get_w(dataMat, labelMat, alphas):  # 获取alphas的值
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot(alphas.T, np.tile(labelMat.reshape(1, -1).T,(1, 2)) * dataMat).T  # 结合公式计算w
    return w.tolist()





if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifer(dataMat, w, b)
