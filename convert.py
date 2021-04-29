import tensorflow as tf
from tensorflow.python.keras.models import load_model
import os

modelName = "初学者位置稳定性_Dense1_训练部分包含70测试_扩容_不固定_0.941__20210428_05_48_37.h5"
modelPath = "./model"
className = "PostionStablity"

model = load_model(os.path.join(modelPath, className, modelName))
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('postion_model.tflite', 'wb') as f:
    f.write(tflite_model)
