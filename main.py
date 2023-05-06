# this script provide a simple implementation of the DBSCAN algorithm

import numpy as np
import matplotlib.pyplot as plt
import random as rd

def euclidien_dist(X, index_i, index_j):
    """calculates the distance between two points in using their indexes"""
    pt1 = X[index_i, :]
    pt2 = X[index_j, :]
    diff = pt1 - pt2
    diff = diff ** 2
    return np.sqrt(diff.sum())


def calc_close_pts_num(X,epsilon):
    res = [-1 for _ in range(len(X))]
    for i in range(len(res)):
        euclidien_dist_to_current_point = []
        for j in range(0,len(res)):
            euclidien_dist_to_current_point.append(euclidien_dist(X,i,j))
        # print(euclidien_dist_to_current_point)
        euclidien_dist_to_current_point = np.array(euclidien_dist_to_current_point)
        res[i] = len(euclidien_dist_to_current_point[euclidien_dist_to_current_point <= epsilon])
    return res


def dbscan(X, epsilon, minPts):
    # first we need to calculate  how many points close to each point
    close_pts = calc_close_pts_num(X,epsilon)
    # we will determine the core-points and non core-points using by comparing the  close points to each point and the minPts hyper param
    core_or_not_info = ['core' if i >= minPts else 'non-core' for i in close_pts]
    # now we will go to each core point and assigne it and it's neighbors to a cluster
    clusters = [-1 for _ in range(len(X))]
    # starting with core point
    random_index = 0
    while core_or_not_info[random_index] == 'non-core':
        random_index = rd.randint(0, len(X)-1)

    cluster_index = 0  # first cluster
    clusters[random_index] = cluster_index
    





if __name__ == "__main__":
    X = np.array(
        [[1, 6], [2, 6.2], [3, 6.3], [2, 5], [3, 5.1], [4, 5.2], [3, 4], [4, 4.2], [5, 4.6], [2, 7], [3, 7], [2, 8],
         [2, 9], [1, 11], [2, 10], [4, 3], [5, 3.2], [6, 3.7], [5, 2], [6, 3], [5, 7], [5, 7.5], [5.5, 7], [5.5, 7.5],
         [5.6, 8], [5, 10], [8, 10], [5, 6]])
    plt.scatter(X[:, 0], X[:, 1])
    plt.title("initial points")
    plt.xticks([i for i in range(1, 20)])
    plt.yticks([i for i in range(1, 20)])
    plt.show()
    dbscan(X, 1.5, 3)
