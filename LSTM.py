from datetime import datetime

from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os

from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from dataReader import load_dataset

modelName = "左右得分_"

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
epochs, batch_size = 300, 32
dataSet = "./data"
className = "zuoyou"
logDir = "./logs"
curTime = datetime.now().strftime("_%Y%m%d_%H_%M_%S")
modelPath = "./model"


def create_model():
    """

    创建 手臂 手腕 模型

    :return: 模型
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=(30, 9), return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation="softmax"))

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0003,
        decay_steps=3000,
        decay_rate=0.8)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['acc'])
    model.summary()
    return model

def create_model2():
    """

    创建 左右 模型

    :return: 模型
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=(30, 9), return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0003,
        decay_steps=3000,
        decay_rate=0.8)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['acc'])
    model.summary()
    return model


def get_callbacks():
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),  # 就是需要对验证集的loss监听
        callbacks.TensorBoard(log_dir=os.path.join(logDir, className, modelName + curTime)),
    ]


def train_model(model, trainX, trainy, testX, testy):
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, validation_split=0.3,
                        callbacks=get_callbacks(), shuffle=True)
    result = model.evaluate(testX, testy, batch_size=batch_size)
    return history, result


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_dataset(dataSet, className,scores=[0,1])

    model = create_model2()

    history, result = train_model(model, X_train, y_train, X_test, y_test)

    saveName = modelName + str(round(result[0], 3)) + "_" + curTime + ".h5"
    model.save(os.path.join(modelPath, className, saveName))

    # plot_history(history)
    # plot_predict(model, X_test, y_test)
    # plt.show()
