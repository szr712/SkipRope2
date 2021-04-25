import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

dirName = "./data/fft"

fileList = os.listdir(dirName)
for file in fileList:
    row_data = pd.read_csv(os.path.join(dirName, file))
    dataframe = row_data.copy()
    # data = dataframe[["accelerationx", "accelerationy", "accelerationz"]]
    data = dataframe[["accelerationy"]]
    plt.plot(data)
    x_time = data.index
    y = data.values
    rate = 25
    y_f = np.fft.fftshift(np.fft.fft(y))
    x_f = np.fft.fftshift(np.fft.fftfreq(x_time.size, d=1 / rate))
    plt.figure()
    plt.plot(x_f, y_f)
    plt.show()
