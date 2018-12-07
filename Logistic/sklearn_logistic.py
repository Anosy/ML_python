'''
使用sklearn中的逻辑回归进行从疝气病症状预测病马的死亡率
sklearn.linear_model.LogisticRegression 在线性模型中有很多的回归方式，如Lasso回归，岭回归等等，本次只使用逻辑回归
class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True,
            intercept_scaling=1, class_weight=None, random_state=None, solver=’liblinear’,
            max_iter=100, multi_class=’ovr’, verbose=0, warm_start=False, n_jobs=1)
参数说明：
    penalty:惩罚项，str类型，可选参数为l1和l2，默认为l2。用于指定惩罚项中使用的规范。目的就是为了防止发生过拟合现象
    dual：对偶或原始方法，bool类型，默认为False。对偶方法只用在求解线性多核(liblinear)的L2惩罚项上。当样本数量>样本特征的时候，dual通常设置为False。
    tol：停止求解的标准，float类型，默认为1e-4。
    fit_intercept：是否存在截距或偏差，bool类型，默认为True。
    **class_weight：用于标示分类模型中各种类型的权重，可以是一个字典或者balanced字符串，默认为不输入，也就是不考虑权重，即为None。
        作用：1.如果误分类代价很高，那么就需要提高不希望被误分类的类的权重，保证其较大程度不被误分类
              2.如果样本存在高度的失衡性，也就是说可能一个类的样本很多，但是另外一个类的样本较少，这时就需要提高少样本的权重，保持其在训练中的地位
    random_state：随机数种子，int类型，可选参数，默认为无，仅在正则化优化算法为sag,liblinear时有用。
    solver：优化算法选择参数，只有五个可选参数，即newton-cg,lbfgs,liblinear,sag,saga。默认为liblinear。
        liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
        lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
        newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
        sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。
        saga：线性收敛的随机优化算法的的变重。
        一般如果样本数量较小的时候选择liblinear就够了，但是如果样本较大，可以考虑使用sag和saga。
    multi_class：分类方式选择参数，str类型，可选参数为ovr和multinomial，默认为ovr。如果选择了ovr那么损失函数的优化方法就可以选择全部，
        但是如果选择为multinomial那么就只能选newton-cg, lbfgs和sag
    n_jobs：并行数。int类型，默认为1
'''
from sklearn.linear_model import LogisticRegression

def colicSklearn():
    frTrain  = open('horseColicTraining.txt','r')
    frTest = open('horseColicTest.txt')
    trainSet = []; trainLabels = []
    testSet = []; testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainSet.append(lineArr)
        trainLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))

    clf = LogisticRegression(solver='liblinear',max_iter=10).fit(trainSet,trainLabels)
    test_accuracy = clf.score(testSet,testLabels)
    print('正确率为%.2f%%' % (test_accuracy*100))



if __name__ == '__main__':
    colicSklearn()