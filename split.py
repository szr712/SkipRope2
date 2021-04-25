import os
import shutil

from sklearn.model_selection import train_test_split

className = "PostionStablity"
dataSet = "./data"
srcPath = "all"

fileList = os.listdir(os.path.join(dataSet, className, srcPath))

trainList, testList = train_test_split(fileList, test_size=0.33)

for file in trainList:
    shutil.copy(os.path.join(dataSet, className, srcPath, file), os.path.join(dataSet, className, "train", file))

for file in testList:
    shutil.copy(os.path.join(dataSet, className, srcPath, file), os.path.join(dataSet, className, "test", file))
