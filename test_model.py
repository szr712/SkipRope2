import os

from sklearn.metrics import classification_report
from tensorflow.python.keras.models import load_model

from dataReader import load_dataset2,load_dataset_beginner

modelName = "手臂得分_class_weight_0.974__20210419_20_14_55.h5"
modelPath = "./model"
className = "shoubi"


def test_classification(model):
    X_train, X_test, y_train, y_test, _ = load_dataset2("./data", className)
    # X_train, X_test, y_train, y_test, _ = load_dataset_beginner("./data", className)

    model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)
    y_pred = y_pred.argmax(axis=1)
    y_test = y_test.argmax(axis=1)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    model = load_model(os.path.join(modelPath, className, modelName))
    model.summary()
    test_classification(model)
