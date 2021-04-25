from sklearn.manifold import TSNE
from tensorflow.python.keras.models import load_model, Model
import numpy as np
from dataReader import load_dataset_beginner
from matplotlib import cm
import matplotlib.pyplot as plt

dataSet = "./data"
className = "PostionStablity"
modelName = "./model\PostionStablity\初学者位置稳定性_Dense1_新数据_不固定_不扩容_取消回滚_0.654__20210425_05_49_26.h5"


def plot_with_labels(lowDWeights, labels, list):
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    i = 0
    for x, y, s, t in zip(X, Y, labels, list):
        c = cm.rainbow(int(255 / 6 * s))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        plt.text(x, y, t.split(".")[0], backgroundcolor=c, fontsize=9)
        i += 1
        print(i)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, class_weights, list = load_dataset_beginner(dataSet, className, augment=False,
                                                                                  times=10)
    model = load_model(modelName)

    for index, layer in enumerate(model.layers):
        print(layer)
        print(index)

    feature = Model(inputs=model.input, outputs=model.layers[71].output)
    a = feature.predict(X_train)
    b = feature.predict(X_test)
    pred = np.vstack((a, b))
    print(pred.shape)

    a = y_train.argmax(axis=1)
    b = y_test.argmax(axis=1)
    b = [x + 3 for x in b]
    lable = np.concatenate((a, b))
    print(lable.shape)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

    low_dim_embs = tsne.fit_transform(pred)

    plot_with_labels(low_dim_embs, lable, list)
