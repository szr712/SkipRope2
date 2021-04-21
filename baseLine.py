from tensorflow.python.keras.models import load_model

model = load_model("./model\zuoyou\左右得分_class_weight_0.997__20210419_19_16_24.h5")

for index, layer in enumerate(model.layers):
    print(index)
    print(layer)
