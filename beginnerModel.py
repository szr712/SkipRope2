from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model

from dataReader import padding


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
    model = Sequential()
    model.add(LSTM(64, input_shape=(30, 9), return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    return model


def zuoyou_model():
    input = Input(shape=(1200, 10))
    # all_input = all_input.numpy()
    # circleList = to_circleList(all_input)
    input=Input(shape=(30,9))
    outs = []

    clayer = circle_layer()

    for i in range(0,5):
        outs.append(clayer(tf.convert_to_tensor(input)))

    concatenated = concatenate(outs)
    out = Dense(1, activation='sigmoid')(concatenated)
    model = Model([all_input,input], out)
    return model


if __name__ == "__main__":
    model = zuoyou_model()
    plot_model(model, to_file='./model.png')
    model.summary()
