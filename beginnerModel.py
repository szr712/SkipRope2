from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model

from dataReader import padding, load_dataset_beginner
from datetime import datetime

modelName = "手臂得分_class_weight_"

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
epochs, batch_size = 300, 32
dataSet = "./data"
className = "PostionStablity"
logDir = "./logs"
curTime = datetime.now().strftime("_%Y%m%d_%H_%M_%S")
modelPath = "./model"


def to_circleList(data):
    hallsensor = -1
    circleList = []
    pre = 0
    for i in range(0, data.shape[0]):
        if data[i][9] == hallsensor:
            hallsensor = -hallsensor
            # 切割进入list
            circleList.append(data[pre:i, 0:9].copy())
            pre = i
    circleList.append(data[pre:, 0:9].copy())
    for i in range(0, len(circleList)):
        circleList[i] = padding(circleList[i])
    return circleList


def circle_layer():
    model = Sequential(name="circle_layer")
    model.add(LSTM(64, input_shape=(30, 9), return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    return model


def zuoyou_model():
    inputs = []
    for i in range(0, 70):
        inputs.append(Input(shape=(30, 9)))

    clayer = circle_layer()

    outs = []

    for input in inputs:
        outs.append(clayer(input))

    concatenated = concatenate(outs)
    out = Dense(1, activation='sigmoid')(concatenated)
    model = Model(inputs, out)
    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset_beginner(dataSet, className)

    model = zuoyou_model()
    # plot_model(model, to_file='./model.png')
    model.summary()

    # 继续编写模型部分代码，需要compile，另外也需要测试目前datareader的写法是否正确,特别是多输入的编写是否正确
    # 另外，class_weight也需要应用在初学者数据集上，等模型运行成功后处理
    # 还需要读入已训练的参数
    # dataReader部分可能需要每个input单独建立array，明天基本写完模型测试时再说
