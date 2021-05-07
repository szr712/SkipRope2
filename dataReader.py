import pickle
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import numpy as np
from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical


def padding(circle):
    """

    对每一圈数据做padding操作
    长于30的圈取后30个数据
    短于30的圈用0填充至30行

    :param circle: 每圈数据
    :return:
    """
    if circle.shape[0] > 30:
        circle = circle[circle.shape[0] - 30:, :]
    else:
        circle = np.pad(circle, ((0, 30 - circle.shape[0]), (0, 0)), 'constant',
                        constant_values=(0.0, 0.0))  # 将数据填充至30行
    return circle.copy()


def to_circleList(data):
    """

    用于教练数据集的分圈
    并取数据前9列

    :param data: 全部数据
    :return: 由各圈组成的list，各圈只保留前9列
    """
    id = data[0, 9]
    pre = 0
    circleList = []
    for i in range(0, data.shape[0]):
        if data[i, 9] != id:
            id = data[i, 9]
            circle = data[pre:i, 0:9].copy()
            pre = i
            circleList.append(circle)
            # print(circle.shape)
    return circleList


def to_circleList_beginner(data):
    """

    用于初学者数据集的分圈
    并取数据前9列

    :param data: 全部数据
    :return: 由各圈组成的list，各圈只保留前9列
    """
    hallsensor = -1
    pre = 0
    circleList = []
    for i in range(0, data.shape[0]):
        if data[i, 10] == hallsensor:
            hallsensor = -hallsensor
            circle = data[pre:i, 0:9].copy()
            pre = i
            circleList.append(circle)
            # print(circle.shape)
    return circleList


def process_circleList(list):
    """

    用于处理初学者模型的circleList

    :param list: 分圈后的list
    :return:
    """
    circleList = list.copy()

    # 如果超过70圈 保留后70圈
    if len(circleList) > 70:
        circleList = circleList[70 - len(circleList):]

    for i, circle in enumerate(circleList):
        circleList[i] = padding(circle)

    # 如果不到70圈用0补足70圈
    zero = np.zeros(shape=(30, 9))
    while len(circleList) < 70:
        circleList.append(zero)

    return circleList


def process_label(score):
    """

    用于处理初学者数据的label，将2分和4分随机加减1

    :param score:
    :return:
    """
    option = [1, -1]
    if score == 2:
        score = score + random.choice(option)
    elif score == 4:
        score = score + random.choice(option)

    return score


def load_file(filepath, isBeginner=True):
    """

    加载数据文件
    并进行数据归一化

    :param filepath:数据文件路径
    :return:
    """
    print(filepath)
    row_data = pd.read_csv(filepath)
    dataframe = row_data.copy()
    # 选用三轴加速度、三轴角加速度、欧拉角、和圈id
    colList = ["accelerationx", "accelerationy", "accelerationz", "angularvelocityx", "angularvelocityy",
               "angularvelocityz", "pitch", "roll", "yaw", "mdlcircle_id", "hallsensor"]
    # sns.pairplot(dataframe[colList], diag_kind="kde")
    # plt.show()
    data = dataframe[colList].values
    data = data.astype(np.float64)
    data[:, 0:6] = data[:, 0:6] / 32768.0  # 对前6列数据归一化
    # if isBeginner:
    #     data[:, 6:9] = data[:, 6:9] / 32768.0  # 对欧拉角归一化
    # else:
    #     data[:, 6:9] = data[:, 6:9] / 180.0  # 对欧拉角归一化
    data[:, 6:9] = data[:, 6:9] / 180.0
    return data


def load_dataset(dirname, classname, scores=[1, 3, 5]):
    """

    加载教练数据集，在线切割测试集与训练集

    :param dirname: 数据集路径
    :param classname: 评价指标
    :param scores: 具体分类类别
    :return:
    """
    X = []
    y = []
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(scores)

    for i, score in enumerate(scores):
        fileList = os.listdir(os.path.join(dirname, classname, str(score)))

        for file in fileList:
            # print(file)
            data = load_file(os.path.join(dirname, classname, str(score), file), isBeginner=False)
            circleList = to_circleList(data.copy())
            for a, circle in enumerate(circleList):
                circleList[a] = padding(circle)
            for circle in circleList:
                # print(circle.shape)
                X.append(circle)
                y.append([encoded[i]])

    X = np.array(X)
    y = np.array(y)

    y = to_categorical(y)

    # 随机分割测试集与训练集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test


def load_dataset2(dirname, classname, scores=[1, 3, 5]):
    """

    加载教练数据集，用于固定测试集与训练集
    并增加返回了用于平衡数据集的class_weights

    :param dirname: 数据集路径
    :param classname: 评价指标
    :param scores: 具体分类类别
    :return:
    """
    X_train = []
    y_train = []
    y2 = []  # 用作class_weight
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(scores)

    fileList = os.listdir(os.path.join(dirname, classname, "train"))

    for file in fileList:
        # print(file)
        data = load_file(os.path.join(dirname, classname, "train", file), isBeginner=False)
        data = data[:, 0:9].copy()  # 取前9列
        X_train.append(padding(data))
        y_train.append([encoded[scores.index(int(file.split("_")[0]))]])
        y2.append(int(file.split("_")[0]))

    # 计算class_weights
    class_weights_array = class_weight.compute_class_weight('balanced', np.unique(scores), np.asarray(y2))
    class_weights = {}
    for i in range(0, encoded.shape[0]):
        class_weights[i] = class_weights_array[i]

    X_train = np.array(X_train)
    y_train = to_categorical(np.array(y_train))  # one-hot编码

    X_test = []
    y_test = []

    fileList = os.listdir(os.path.join(dirname, classname, "test"))

    for file in fileList:
        data = load_file(os.path.join(dirname, classname, "test", file), isBeginner=False)
        data = data[:, 0:9].copy()
        X_test.append(padding(data))
        y_test.append([encoded[scores.index(int(file.split("_")[0]))]])

    X_test = np.array(X_test)
    y_test = to_categorical(np.array(y_test))  # one-hot编码

    return X_train, X_test, y_train, y_test, class_weights


def load_dataset_beginner(dirname, classname, pklPath="./data/pkl", augment=False, times=200):
    X_train = [[] for _ in range(70)]
    y_train = []
    y2 = []

    scores = [1, 3, 5]
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(scores)

    trainList = os.listdir(os.path.join(dirname, classname, "train"))
    testList = os.listdir(os.path.join(dirname, classname, "test"))

    with open(os.path.join(pklPath, "index_2_" + classname + ".pkl"), 'rb') as f:
        index_2_label = pickle.load(f, encoding='bytes')

    for file in trainList:
        data = load_file(os.path.join(dirname, classname, "train", file))
        circleList = to_circleList_beginner(data)
        if not augment:
            circleList = process_circleList(circleList)
            for i, circle in enumerate(circleList):
                X_train[i].append(circle)
            y_train.append(encoded[scores.index(process_label(int(index_2_label[int(file.split(".")[0])])))])
            y2.append(process_label(int(index_2_label[int(file.split(".")[0])])))
        else:
            # 在线扩容
            for i in range(0, times):
                if i != 0:
                    random.shuffle(circleList)
                list = process_circleList(circleList)
                for i, circle in enumerate(list):
                    X_train[i].append(circle.copy())

                y_train.append(encoded[scores.index(process_label(int(index_2_label[int(file.split(".")[0])])))])
                y2.append(process_label(int(index_2_label[int(file.split(".")[0])])))

    # 计算class_weights
    class_weights_array = class_weight.compute_class_weight('balanced', np.unique(scores), np.asarray(y2))
    class_weights = {}
    for i in range(0, encoded.shape[0]):
        class_weights[i] = class_weights_array[i]

    for i, x in enumerate(X_train):
        X_train[i] = np.array(x)
    y_train = to_categorical(np.array(y_train))  # one-hot编码

    X_test = [[] for _ in range(70)]
    y_test = []

    for file in testList:
        data = load_file(os.path.join(dirname, classname, "test", file))
        circleList = to_circleList_beginner(data)

        # 在线扩容
        # for i in range(0, 200):
        #     if i != 0:
        #         random.shuffle(circleList)
        #     list = process_circleList(circleList)
        #     for i, circle in enumerate(list):
        #         X_test[i].append(circle.copy())
        #
        #     y_test.append(encoded[scores.index(process_label(int(index_2_label[int(file.split(".")[0])])))])
        circleList = process_circleList(circleList)
        for i, circle in enumerate(circleList):
            X_test[i].append(circle)
        y_test.append(encoded[scores.index(process_label(int(index_2_label[int(file.split(".")[0])])))])

    for i, x in enumerate(X_test):
        X_test[i] = np.array(x)
    y_test = to_categorical(np.array(y_test))  # one-hot编码

    return X_train, X_test, y_train, y_test, class_weights, trainList + testList


def load_dataset_beginner_reg(dirname, classname, pklPath="./data/pkl"):
    X_train = [[] for _ in range(70)]
    y_train = []

    # 定义gauss噪声的均值和方差
    mu = 0
    sigma = 0.25

    trainList = os.listdir(os.path.join(dirname, classname, "train"))
    testList = os.listdir(os.path.join(dirname, classname, "test"))

    with open(os.path.join(pklPath, "index_2_" + classname + ".pkl"), 'rb') as f:
        index_2_label = pickle.load(f, encoding='bytes')

    for file in trainList:
        data = load_file(os.path.join(dirname, classname, "train", file))
        circleList = to_circleList_beginner(data)
        circleList = process_circleList(circleList)
        for i, circle in enumerate(circleList):
            X_train[i].append(circle)

        # # 在线扩容
        # for i in range(0, 200):
        #     if i != 0:
        #         random.shuffle(circleList)
        #     list = process_circleList(circleList)
        #     for i, circle in enumerate(list):
        #         X_train[i].append(circle.copy())
        #
        #     y_train.append(int(index_2_label[int(file.split(".")[0])]))
        y_train.append(int(index_2_label[int(file.split(".")[0])]))

    # 计算class_weights
    # class_weights_array = class_weight.compute_class_weight('balanced', np.unique(scores), np.asarray(y_train))
    # class_weights = {}
    # for i, score in enumerate(scores):
    #     class_weights[score] = class_weights_array[i]

    for i, x in enumerate(X_train):
        X_train[i] = np.array(x)
    y_train = np.array(y_train, dtype=np.float64)

    X_test = [[] for _ in range(70)]
    y_test = []

    for file in testList:
        data = load_file(os.path.join(dirname, classname, "test", file))
        circleList = to_circleList_beginner(data)

        # 在线扩容
        # for i in range(0, 200):
        #     if i != 0:
        #         random.shuffle(circleList)
        #     list = process_circleList(circleList)
        #     for i, circle in enumerate(list):
        #         X_test[i].append(circle.copy())
        #
        #     y_test.append(encoded[scores.index(process_label(int(index_2_label[int(file.split(".")[0])])))])
        circleList = process_circleList(circleList)
        for i, circle in enumerate(circleList):
            X_test[i].append(circle)
        y_test.append(int(index_2_label[int(file.split(".")[0])]))

    for i, x in enumerate(X_test):
        X_test[i] = np.array(x)
    y_test = np.array(y_test, dtype=np.float64)

    # 对训练集标签增加高斯噪声
    for i in range(y_train.shape[0]):
        y_train[i] += random.gauss(mu, sigma)
    # plt.scatter(np.arange(y_train.shape[0]), y_train)
    # plt.show()

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # X_train, X_test, y_train, y_test, class_weight = load_dataset2("./data", "shouwan")
    X_train, X_test, y_train, y_test, class_weights, _ = load_dataset_beginner("./data", "PostionStablity",
                                                                               augment=True)
    # print(X_train.shape)
    # print(X_test.shape)
    print(len(X_train))
    print(len(X_test))
    for x in X_train:
        print(x.shape)
    for x in X_test:
        print(x.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(class_weights)
