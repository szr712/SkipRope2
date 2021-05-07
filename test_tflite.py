import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import classification_report

from dataReader import load_dataset_beginner

modelName = "rope_model_hq.tflite"
modelPath = "./model"
className = "RopeSwinging"

X_train, X_test, y_train, y_test, _, list = load_dataset_beginner("./data", className)
size = X_test[0].shape[0]

y_pred = []

# interpreter = tf.lite.Interpreter(model_path=os.path.join(modelPath, className, modelName))
interpreter = tf.lite.Interpreter(model_path=modelName)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for i, data in enumerate(X_test):
    # X_test[i]=X_test[i].astype(np.float32)
    X_test[i] = data.tolist()

for i in range(0, size):
    for input, data in zip(input_details, X_test):
        inpue_data = np.array(data[i])
        inpue_data = [inpue_data.astype(np.float32)]
        interpreter.set_tensor(input['index'], inpue_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    y_pred.append(output_data.argmax(axis=1))

    # print(output_data)

print(y_pred)
y_pred = np.ravel(np.vstack(y_pred))
y_test = y_test.argmax(axis=1)

print(classification_report(y_test, y_pred))

print(input_details)
print(output_details)
