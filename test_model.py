import os

import pickle
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.models import load_model

from dataReader import load_dataset2, load_dataset_beginner
import pandas as pd
import seaborn as sns


modelName = "初学者动作标准度_Dense1_训练包含75测试_扩容_不固定_0.898__20210428_08_33_22.h5"
modelPath = "./model"
className = "RopeSwinging"
pklPath = "./data/pkl"


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

    for y, index in zip(y_true.tolist() + y_pred.tolist(), list):
        print(y, index, index_2_label[int(index.split(".")[0])])

    print(classification_report(y_test, y_pred))

    # C = confusion_matrix(y_test, y_pred, labels=["1", "3", "5"])
    # df = pd.DataFrame(C, index=["1", "3", "5"], columns=["1", "3", "5"])
    # sns.heatmap(df, annot=True)


if __name__ == "__main__":
    model = load_model(os.path.join(modelPath, className, modelName))
    model.summary()
    test_classification(model)
