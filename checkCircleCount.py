import os
from collections import Counter

import numpy as np

from dataReader import load_file, to_circleList_beginner

dataSet = "./data"
className = "SpeedStability"

trainList = os.listdir(os.path.join(dataSet, className, "train"))
testList = os.listdir(os.path.join(dataSet, className, "test"))

countList = []

for file in trainList:
    data = load_file(os.path.join(os.path.join(dataSet, className, "train", file)))
    circleList = to_circleList_beginner(data)
    countList.append(len(circleList))

for file in testList:
    data = load_file(os.path.join(os.path.join(dataSet, className, "test", file)))
    circleList = to_circleList_beginner(data)
    countList.append(len(circleList))

print(countList)
print(np.mean(countList))
print(Counter(countList))
print(np.quantile(countList, 0.75))
print(np.median(countList))
