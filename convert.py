import tensorflow as tf
from tensorflow.python.keras.models import load_model, Model
import os

modelName = "初学者位置稳定性_Dense1_训练部分包含70测试_扩容_不固定_0.941__20210428_05_48_37.h5"
modelPath = "./model"
className = "PostionStablity"


def rename_input(model):
    for layer, i in zip(model.layers, range(1, 71)):
        layer._name = "my_input_"+str(i)


def convert_to_tflite_model(model, save_file_path, conversion_mode="normal"):
    """
    model:
        normal tf keras model
    save_file_path:
        path including filename for output tflite model
    conversion_mode:
        tflite conversion mode. can be one of ["normal", "fp16_quantization", "hybrid_quantization"]
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if conversion_mode == "normal":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    elif conversion_mode == "fp16_quantization":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
    elif conversion_mode == "hybrid_quantization":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.experimental_new_converter = True
    else:
        raise Exception("`%s` is not supported conversion_mode" % conversion_mode)

    tflite_model = converter.convert()
    open(save_file_path, "wb").write(tflite_model)


if __name__ == "__main__":
    model = load_model(os.path.join(modelPath, className, modelName))
    model.summary()

    rename_input(model)

    # for index, layer in enumerate(model.layers):
    #     print(index)
    #     print(layer)

    convert_to_tflite_model(model,'postion_model_fp16.tflite',conversion_mode="fp16_quantization")

    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    #
    # # Save the model.
    # with open('postion_model.tflite', 'wb') as f:
    #     f.write(tflite_model)
