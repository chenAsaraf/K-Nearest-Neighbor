import DataControl
import numpy as np


def lp_distace(p, vec1, vec2):
    diff = np.subtract(vec1, vec2)
    diff = np.absolute(diff)
    if p == float('inf'):
        return np.amax(diff)
    else:
        diffPow = np.power(diff, p)
        sumDiff = np.sum(diffPow)
        return np.power(sumDiff, 1/p)

def knn_search(test_features, base_features, base_labels, p, k):
    # label each new point by the k nearest neighbor
    test_results = np.empty([test_features.shape[0]])
    for point in range(test_features.shape[0]):
        k_nearest_dist = np.full((k), np.inf) # array of distances to the nearest neighbors
        k_labels = np.empty([k]) # The labels of the nearest neighbors
        for index in range(base_features.shape[0]):
            # compute the difference between the test-point to the base-point by lp distance:
            dif = lp_distace(p, test_features[point], base_features[index])
            # see if it is closer than a neighbor already exists:
            if dif < np.amax(k_nearest_dist):
                # put the shorter distance into in the max distance place
                current_max_index = np.argmax(k_nearest_dist)
                k_nearest_dist[current_max_index] = dif
                k_labels[current_max_index] = base_labels[index]
        # determine the label of the new point as voted by most neighbors
        vote = np.sum(k_labels)
        if vote < 0:
            test_results[point] = -1
        else:
            test_results[point] = 1
    return test_results






if __name__ == '__main__':
    # initiailize data set
    path = 'HC_Body_Tempeature.txt'
    data = DataControl.read_data(path)
    features, labels = DataControl.splitToFeaturesLabels(data)

    k = 3
    p = 1
    base_features = np.array([[0,0],
                              [1,1],
                              [2,0.5],
                              [2,2]])
    base_labels = np.array([1, 1, -1, -1])

    test_features = np.array([[1,2],
                              [0,2]])

    result = knn_search(test_features, base_features, base_labels, p, k)

    print(result)


