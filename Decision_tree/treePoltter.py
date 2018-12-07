import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle='sawtooth', fc='0.8')  # 定义决策节点，boxstyle指的是文本框的类型，sanwtooth是锯齿形，fc是边框线的粗细
leafNode = dict(boxstyle='round4', fc='0.8')  # 定义决策树的叶子节点的描述属性
arrow_args = dict(arrowstyle='<-')  # 定义决策树的箭头属性


# 绘制带箭头的注解
# nodeTxt：节点的文字标注, centerPt：节点中心位置,
# parentPt：箭头起点位置（上一节点位置）, nodeType：节点属性
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def getNumLeafs(myTree):   # 获取叶子节点的数量
    numLeafs = 0
    firstStr = myTree.keys()
    for each in firstStr:   # 由于获取的firstStr，类型为字典的键类型，使用此方法来进行转换
        Str = each
    secondDict = myTree[Str]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])  # 此处使用了迭代方法计算叶节点
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()
    for each in firstStr:  # 由于获取的firstStr，类型为字典的键类型，使用此方法来进行转换
        Str = each
    secondDict = myTree[Str]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:  # 获取最大层数
            maxDepth = thisDepth
    return maxDepth


def plotMidText(cntrPt, parentPt, txtString):  # 在父节点和子节点之间添加文本
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]   # 文本的x位置
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]   # 文本的y位置
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree,parentPt,nodeTxt):    # 绘制决策树
    numLeafs = getNumLeafs(myTree)   # 计算树的叶节点的数量
    depth = getTreeDepth(myTree)   # 计算树的深度
    firstStr = myTree.keys()
    for each in firstStr:  # 由于获取的firstStr，类型为字典的键类型，使用此方法来进行转换
        Str = each
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)  # 获取叶节点的中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)    # 在父节点和子节点之间添加文本
    plotNode(Str, cntrPt, parentPt, decisionNode)  # 绘制每个节点
    secondDict = myTree[Str]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):   # 创建决策树的主函数
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))   # 获取宽度
    plotTree.totalD = float(getTreeDepth(inTree))   # 获取深度
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
