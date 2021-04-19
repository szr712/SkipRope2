import os
import pandas as pd
from sklearn.model_selection import train_test_split

from dataReader import to_circleList

dataSet = "./data"
className = "zuoyou"
scores = [0, 1]

for score in scores:
    allCircle = []
    fileList = os.listdir(os.path.join(dataSet, className, str(score)))
    for file in fileList:
        row_data = pd.read_csv(os.path.join(os.path.join(dataSet, className, str(score), file)))
        dataframe = row_data.copy()
        colList = ["accelerationx", "accelerationy", "accelerationz", "angularvelocityx", "angularvelocityy",
                   "angularvelocityz", "pitch", "roll", "yaw", "mdlcircle_id"]
        data = dataframe[colList].values
        circleList = to_circleList(data)

        pre = 0
        for circle in circleList:
            allCircle.append(dataframe[pre:pre + circle.shape[0]])
            pre = pre + circle.shape[0]

    trainList, testList = train_test_split(allCircle, test_size=0.33)
    for i, circle in enumerate(trainList):
        circle.to_csv(os.path.join(dataSet, className, "train", str(score) + "_" + str(i) + ".csv"), index=False)
    for i, circle in enumerate(testList):
        circle.to_csv(os.path.join(dataSet, className, "test", str(score) + "_" + str(i) + ".csv"), index=False)
