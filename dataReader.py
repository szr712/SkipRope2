import pandas as pd
import os
import numpy as np
from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical


def padding(circle):
    if circle.shape[0] > 30:
        circle = circle[circle.shape[0] - 30:, :]
    else:
        circle = np.pad(circle, ((0, 30 - circle.shape[0]), (0, 0)), 'constant',
                        constant_values=(0.0, 0.0))  # 将数据填充至30行
    return circle.copy()


def to_circleList(data):
    id = data[0, 9]
    pre = 0
    circleList = []
    for i in range(0, data.shape[0]):
        if data[i, 9] != id:
            id = data[i, 9]
            circle = data[pre:i, 0:9].copy()
            pre = i
            circleList.append(circle)
            print(circle.shape)
    return circleList


def load_file(filepath):
    print(filepath)
    row_data = pd.read_csv(filepath)
    dataframe = row_data.copy()
    # 选用三轴加速度、三轴角加速度、欧拉角、和圈id
    colList = ["accelerationx", "accelerationy", "accelerationz", "angularvelocityx", "angularvelocityy",
               "angularvelocityz", "pitch", "roll", "yaw", "mdlcircle_id"]
    # sns.pairplot(dataframe[colList], diag_kind="kde")
    # plt.show()
    data = dataframe[colList].values
    data = data.astype(np.float64)
    data[:, 0:6] = data[:, 0:6] / 32768.0  # 对前6列数据归一化
    data[:, 6:9] = data[:, 6:9] / 180.0  # 对欧拉角归一化
    return data


def load_dataset(dirname, classname, scores=[1, 3, 5]):
    X = []
    y = []
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(scores)

    for i, score in enumerate(scores):
        fileList = os.listdir(os.path.join(dirname, classname, str(score)))

        for file in fileList:
            # print(file)
            data = load_file(os.path.join(dirname, classname, str(score), file))
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
    X_train = []
    y_train = []
    y2 = []  # 用作class_weight
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(scores)

    fileList = os.listdir(os.path.join(dirname, classname, "train"))

    for file in fileList:
        # print(file)
        data = load_file(os.path.join(dirname, classname, "train", file))
        data = data[:, 0:9].copy()
        X_train.append(padding(data))
        y_train.append([encoded[scores.index(int(file.split("_")[0]))]])
        y2.append(int(file.split("_")[0]))

    class_weights_array = class_weight.compute_class_weight('balanced', np.unique(scores), np.asarray(y2))
    class_weights = {}
    for i in range(0, encoded.shape[0]):
        class_weights[i] = class_weights_array[i]

    X_train = np.array(X_train)
    y_train = to_categorical(np.array(y_train))

    X_test = []
    y_test = []

    fileList = os.listdir(os.path.join(dirname, classname, "test"))

    for file in fileList:
        data = load_file(os.path.join(dirname, classname, "test", file))
        data = data[:, 0:9].copy()
        X_test.append(padding(data))
        y_test.append([encoded[scores.index(int(file.split("_")[0]))]])

    X_test = np.array(X_test)
    y_test = to_categorical(np.array(y_test))

    return X_train, X_test, y_train, y_test, class_weights


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, class_weight = load_dataset2("./data", "shouwan")
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(class_weight)
