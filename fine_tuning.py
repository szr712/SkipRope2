from datetime import datetime
import os

from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.version_utils import callbacks

from dataReader import load_dataset_beginner

modelName="初学者位置稳定性_dense1_fine_tuning2_"

epochs, batch_size = 200, 64
dataSet = "./data"
className = "PostionStablity"
logDir = "./logs"
curTime = datetime.now().strftime("_%Y%m%d_%H_%M_%S")
modelPath = "./model"
preModel = "初学者位置稳定性_dense1_0.606__20210422_08_46_52.h5"

def get_callbacks():
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),  # 就是需要对验证集的loss监听
        callbacks.TensorBoard(log_dir=os.path.join(logDir, className, modelName + curTime)),
    ]


def train_model(model, trainX, trainy, testX, testy, class_weights):
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, validation_data=(testX, testy),
                        class_weight=class_weights, callbacks=get_callbacks(), shuffle=True)
    result = model.evaluate(testX, testy, batch_size=batch_size)
    return history, result

def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])
    model.summary()

if __name__=="__main__":

    model = load_model(os.path.join(modelPath, className, preModel))
    extractor = model.get_layer(index=70)
    extractor.summary()
    extractor.trainable = True
    compile_model(model)

    X_train, X_test, y_train, y_test, class_weights = load_dataset_beginner(dataSet, className)
    history, result = train_model(model, X_train, y_train, X_test, y_test, class_weights)

    saveName = modelName + str(round(result[1], 3)) + "_" + curTime + ".h5"
    model.save(os.path.join(modelPath, className, saveName))

