'''
基于sklearn中的贝叶斯，进行新浪新闻的分类，由于新浪新闻的结果分类为多个类，所以选择多项式分布分类器
class sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)  多项式发布贝叶斯分类器，同时还有其他类型的分类器
参数:   alpha：浮点型可选参数，默认为1.0，其实就是添加拉普拉斯平滑，即为上述公式中的λ ，如果这个参数设置为0，就是不添加平滑；
       fit_prior：布尔型可选参数，默认为True。布尔参数fit_prior表示是否要考虑先验概率，如果是false,则所有的样本类别输出都有相同的类别先验概率。
       否则可以自己用第三个参数class_prior输入先验概率，或者不输入第三个参数class_prior让MultinomialNB自己从训练集样本来计算先验概率，
       此时的先验概率为P(Y=Ck)=mk/m。其中m为训练集样本总数量，mk为输出为第k类别的训练集样本数。
       class_prior：可选参数，默认为None。
'''
import os
import jieba  # 中文分词组件，GitHub介绍 https://github.com/fxsjy/jieba
import random
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np


def TextProcessing(folder_path, test_size=0.2):  # 读取文件并且使用jieba将文件解析切分，并且给出文件对应的类
    folder_list = os.listdir(folder_path)  # 查看folder_path下的文件
    data_list = []  # 训练集
    class_list = []

    # 遍历每个子文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)  # 在父文件夹索引下，添加子文件夹目录
        files = os.listdir(new_folder_path)  # 存放子文件夹下的txt文件的列表
        j = 1
        for file in files:
            if j > 100:  # 每类txt样本数最多100个
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:  # 打开txt文件
                raw = f.read()

            word_list = jieba.lcut(raw, cut_all=False)  # 精简模式，返回一个列表，使用cut则返回的为一个迭代器
            # word_list = list(word_cut)  # generator转换为list

            data_list.append(word_list)
            class_list.append(folder)
            j += 1

    data_class_list = list(zip(data_list, class_list))  # zip压缩合并，将数据与标签对应压缩,即将数据和对对应的标签用元组形式保存
    # print(data_class_list)
    random.shuffle(data_class_list)  # 将data_class_list乱序
    # index = int(len(data_class_list) * test_size) + 1  # 训练集和测试集切分的索引值，默认设置test_size=0.2，表示取20%作为测试集
    index = int(len(data_class_list) * test_size)  # 训练集和测试集切分的索引值，默认设置test_size=0.2，表示取20%作为测试集
    train_list = data_class_list[index:]  # 训练集
    test_list = data_class_list[:index]  # 测试集
    train_data_list, train_class_list = zip(*train_list)  # 训练集解压缩
    test_data_list, test_class_list = zip(*test_list)  # 测试集解压缩

    all_words_dict = {}  # 统计训练集词频
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # 根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # lambda为匿名函数作用为了创建函数用完丢弃
    # print(all_words_tuple_list)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩,获取词条的名字，以及词条的数量,且都返回为tuple格式
    all_words_list = list(all_words_list)  # 转换成列表
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def MakeWordsSet(words_file):  # 读取stopword文档，且以set格式来保存。使用set为了除去重复词。
    words_set = set()  # 创建set集合
    with open(words_file, 'r', encoding='utf-8') as f:  # 打开文件
        for line in f.readlines():  # 一行一行读取
            word = line.strip()  # 去回车
            if len(word) > 0:  # 有文本，则添加到words_set中
                words_set.add(word)
    return words_set


def TextFeatures(train_data_list, test_data_list, feature_words):  # 将训练集和测试集向量话，如果对应位置出现feature_words中从词语，则该词位置置1
    def text_features(text, feature_words):  # 出现在特征集中，则置1
        text_words = set(text)
        features = []
        features = [1 if word in text_words else 0 for word in feature_words]
        '''
        for word in feature_words:
            if word in text_words:
                features.append(1)
            else:
                features.append(0)
        '''
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    # print(train_feature_list)
    return train_feature_list, test_feature_list  # 返回结果


def words_dict(all_words_list, deleteN, stopwords_set=set()):  # deleteN:要删除列表前N个无用字符
    feature_words = []  # 特征列表
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # feature_words的维度为1000
            break
        # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):  # 使用贝叶斯进行模型训练和测试
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy


def choose_deleteN():  # 绘制图片，选择使得精确度最高的deleteN
    test_accuracy_list = []
    deleteNs = range(0, 1000, 20)  # 0 20 40 60 ... 980
    for deleteN in deleteNs:  # 每隔20个点，设置deleteN，测试去除不同的开头词得到的训练效果
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list.append(test_accuracy)

    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()


def handle_file(filename):  # 处理需要分类的文件
    f = open(filename, 'r', encoding='utf-8')
    raw = f.read()
    file_list = jieba.lcut(raw, cut_all=False)
    file_dict = {}
    feature_file = []
    for word in file_list:
        if word in file_dict.keys():
            file_dict[word] += 1
        else:
            file_dict[word] = 1
    file_final_list = sorted(file_dict.items(),key=lambda f:f[1],reverse=True)
    feature_file_test, file_final_list_num = zip(*file_final_list)
    feature_file_test = list(feature_file_test)
    for t in range(8,len(feature_file_test)):
        if not feature_file_test[t].isdigit() and 1 < len(feature_file_test[t]) < 5:
            feature_file.append(feature_file_test[t])
    feature_list = []
    for word in feature_words:
        if word in file_list:
            feature_list.append(1)
        else:
            feature_list.append(0)
    feature_list = np.array(feature_list).reshape(1, -1)  # 将列表给向量化，reshape(1,-1)表示转化为一行，但是多少列由系统设定
    return feature_list


if __name__ == '__main__':
    for each in range(10):
        order_dict = {'C000008': '财经', 'C000010': 'IT', 'C000013': '健康', 'C000014': '体育',
                      'C000016': '旅游', 'C000020': '教育', 'C000022': '招聘', 'C000023':'文化','C000024':'军事'}
        # 文本预处理
        folder_path = './SogouC/Sample'  # 训练集存放地址
        all_words_list, train_data_list, test_data_list, \
        train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
        # 生成stopwords_set
        stopwords_file = './stopwords_cn.txt'
        stopwords_set = MakeWordsSet(stopwords_file)  # 获取'停止词'集合
        feature_words = words_dict(all_words_list, 500, stopwords_set)  # 通过观察，发现deleteN=500效果不错
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
        # test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
        feature_list = handle_file('./SogouC/class22.txt')
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        predict = classifier.predict(feature_list)
        print(order_dict[predict[0]])
