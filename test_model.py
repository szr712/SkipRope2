import os

import pickle
from sklearn.metrics import classification_report
from tensorflow.python.keras.models import load_model

from dataReader import load_dataset2, load_dataset_beginner

modelName = "初学者位置稳定性_Dense1_重新划分整体的数据集_扩容_不固定_0.667__20210428_02_07_45.h5"
modelPath = "./model"
className = "PostionStablity"
pklPath="./data/pkl"


def test_classification(model):
    # X_train, X_test, y_train, y_test, _ = load_dataset2("./data", className)
    X_train, X_test, y_train, y_test, _, list = load_dataset_beginner("./data", className)

    model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)
    y_pred = y_pred.argmax(axis=1)
    y_test = y_test.argmax(axis=1)

    y_true = model.predict(X_train).argmax(axis=1)

    with open(os.path.join(pklPath, "index_2_" + className + ".pkl"), 'rb') as f:
        index_2_label = pickle.load(f, encoding='bytes')

    for y, index in zip(y_true.tolist()+y_pred.tolist(), list):
        print(y, index,index_2_label[int(index.split(".")[0])])

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    model = load_model(os.path.join(modelPath, className, modelName))
    model.summary()
    test_classification(model)
