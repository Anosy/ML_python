"""
使用sklearn库，调用决策树对隐形眼镜类型进行预测
 DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1,
 min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,....)
 criterion:特征选择标准，可选参数，默认是gini，可以设置为entropy(香农熵)。ID3算法用的是香农，而CART用的是gini
 splitter：特征划分点选择标准，可选参数，默认是best，可以设置为random。best参数是根据算法选择最佳的切分特征，random随机的在部分划分点中找局部最优的划分点。但是在大数据量的时候还是选择random
 max_features：划分时考虑的最大特征数，可选参数，默认是None。就是对最大特征数量进行筛选，一般特征少的用None即可。
 max_depth：决策树最大深，可选参数，默认是None。这个参数是这是树的层数的。
 min_samples_split：内部节点再划分所需最小样本数，可选参数，默认是2。也就是说，如果样本的个数少于设置的值，那么节点将停止切分。
 max_leaf_nodes：最大叶子节点数，可选参数，默认是None。
 random_state：可选参数，默认是None。如果设置了随机数种子，那么相同随机数种子，不同时刻产生的随机数也是相同的。而如果未设置，那么结果将会随着时间的变化而变化
 Methods:
     fit(X, y[, sample_weight, check_input, ...])	Build a decision tree classifier from the training set (X, y).
     predict(X[, check_input])	Predict class or regression value for X.
     predict_proba(X[, check_input]) Predict class probabilities of the input samples X.
     Returns the mean accuracy on the given test data and labels.
"""
from sklearn import tree
import pandas as pd  # 导入pandas库，进行pandas处理
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # 将字符串转换为增量值，就是将不连续的数字或者文本进行编号  OneHotEncoder就是将非连续给连续化
import pydotplus  # 导入后，调用graph = pydotplus.graph_from_dot_data(dot_data)，可以将图像进行打包
from sklearn.externals.six import StringIO  # StringIO经常被用来作为字符串的缓存，其操作可以像文件一样

if __name__ == '__main__':
    with open('lenses.txt', 'r') as fr:  # 加载文件
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]  # 处理文件
    # print(lenses)
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])  # 获取每个组的类别
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 特征的标签值
    lenses_list = []  # 保存lenses数据的临时列表
    lenses_dict = {}  # 保存lenses数据的字典，用于生成pandas
    for each_label in lensesLabels:  # 提取信息，生成字典
        for each in lenses:  # 取lenses中内嵌的每一个list，循环中完成了，对不同标签值(数据每一列)构成一个list
            lenses_list.append(each[lensesLabels.index(each_label)])  # list.index(obj)返回list中obj的索引位置
            # print(lenses_list)
        lenses_dict[each_label] = lenses_list  # 将获取的list，作为lenses_dict的value值
        lenses_list = []  # 清空lenses_list,保证能够对其他列进行读取
    # print(lenses_dict)  # 打印字典信息
    lenses_pd = pd.DataFrame(lenses_dict)  # 以字典的形式构建二维pandas数据表，上边部分为标签值，下边为表格，且一一对应，前面的序列为自动生产的
    print(lenses_pd)
    le = LabelEncoder()
    for col in lenses_pd.columns:  # 为每一列序列化,lenses_pd[col]相当于取每一列
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # print(lenses_pd)


    clf = tree.DecisionTreeClassifier(max_depth=4)  # 创建DecisionTreeClassifier()类，且定制最大的深度为4
    clf = clf.fit(lenses_pd.values.tolist(),lenses_target)  # 使用数据构建决策树。由于lenses_pd.value为np.ndarray型，因此利用tolist()方法将其转为list格式
    # print(type(lenses_pd.values.tolist())
    # print(clf.class_)
    # print(lenses_pd.keys())
    # dot_data = StringIO()
    dot_data = tree.export_graphviz(decision_tree=clf, out_file=None,
                                    # 绘制决策树, decision_tree:决策树分类器  out_file：输出文件的句柄或名称
                                    feature_names=lenses_pd.keys(),  # feature_names:  特征的名称
                                    class_names=clf.classes_,  # class_names类别的名称
                                    filled=True, rounded=True,
                                    # filled:绘制节点以指示分类的多数类别(以不同颜色表示不同的类)   rounded:绘制具有圆角的节点框,并使用Helvetica字体
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("my_tree.png")  # 也可以是write_pdf('tree.pdf')
    print(clf.predict([[0, 0, 1, 1]]))  # 预测结果
