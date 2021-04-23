import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.python.keras.utils.version_utils import callbacks
from tensorflow.python.keras.utils.vis_utils import plot_model

from dataReader import padding, load_dataset_beginner, load_dataset_beginner_reg
from datetime import datetime

modelName = "初学者位置稳定性_Dense1_分类_不扩容"

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
epochs, batch_size = 200, 64
dataSet = "./data"
className = "PostionStablity"
logDir = "./logs"
curTime = datetime.now().strftime("_%Y%m%d_%H_%M_%S")
modelPath = "./model"
extractor = "zuoyou/左右得分_class_weight_0.997__20210419_19_16_24.h5"


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


def circle_model():
    # model = load_model(os.path.join(modelPath, extractor))
    # model = Model(inputs=model.input, outputs=model.layers[1].output, name="circle_model")
    model = Sequential(name="circle_model")
    model.add(LSTM(64, input_shape=(30, 9), return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    return model


def get_callbacks():
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),  # 就是需要对验证集的loss监听
        callbacks.TensorBoard(log_dir=os.path.join(logDir, className, modelName + curTime)),
    ]


def my_loss_fn(y_true, y_pred):
    zeros = tf.zeros_like(y_pred, dtype=y_pred.dtype)
    ones = tf.ones_like(y_pred, dtype=y_pred.dtype)
    filter = tf.where(tf.abs(y_true - y_pred) > 1, ones, zeros)
    return tf.reduce_mean(filter * tf.square(y_pred - y_true), axis=-1)


def zuoyou_model():
    inputs = []
    for i in range(0, 70):
        inputs.append(Input(shape=(30, 9)))
    print("inputs complicated")

    feature = circle_model()
    # feature.trainable = False

    outs = []

    for input in inputs:
        outs.append(feature(input))
    print("outs complicated")

    x = concatenate(outs)

    # x = Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    # x = tf.expand_dims(x, axis=-1)
    # x = LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    # x = LSTM(96, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    out = Dense(3, activation='softmax')(x)
    # out = Dense(1)(x)
    model = Model(inputs, out)
    return model


def compile_model(model):
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0003,
        decay_steps=3000,
        decay_rate=0.8)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['acc'])
    # model.compile(loss='mse', optimizer=RMSprop(learning_rate), metrics=['mse', 'mae'])
    # model.compile(loss=my_loss_fn, optimizer=RMSprop(learning_rate), metrics=my_loss_fn)
    model.summary()
    return model


def train_model(model, trainX, trainy, testX, testy, class_weights):
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, validation_data=(testX, testy),
                        class_weight=class_weights, callbacks=get_callbacks(), shuffle=True)
    # history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, validation_data=(testX, testy),
    #                     callbacks=get_callbacks(), shuffle=True)
    result = model.evaluate(testX, testy, batch_size=batch_size)
    return history, result


if __name__ == "__main__":
    print(modelName)
    X_train, X_test, y_train, y_test, class_weights = load_dataset_beginner(dataSet, className)

    model = zuoyou_model()
    compile_model(model)
    # plot_model(model, to_file='./model.png')

    history, result = train_model(model, X_train, y_train, X_test, y_test, class_weights)

    saveName = modelName + str(round(result[1], 3)) + "_" + curTime + ".h5"
    model.save(os.path.join(modelPath, className, saveName))
