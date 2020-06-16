import numpy as np


"""
Read data from Hope College data set Temperature.
Note: The first column is Body temperature in degrees Fahrenheit,
      the second is Gender (1 = male, 2 = female)
      and the third is Heart rate in beats per minute
"""
def read_data(path):
    file = np.loadtxt(path)
    file = np.where(file == 2, -1, file)
    x1 = np.array(file[:, 0])
    x2 = np.array(file[:, 2])
    y = np.array(file[:, 1])
    data = np.vstack((x1, x2, y)).T
    return data


"""
In the experiment we randomly divide the data-points for each iteration 
into 65 training points and 65 test points
"""
def splitTestTrain(data):
    np.random.shuffle(data)
    train = data[: 65, :]
    test = data[66:, :]
    return train, test


"""
Split into 2 columns of features (Body temperature, Heart rate)
and one vector of labels (gender)
"""
def splitToFeaturesLabels(data):
    X = np.array(data[:, 0:2])
    Y = np.array(data[:, -1])
    return X, Y