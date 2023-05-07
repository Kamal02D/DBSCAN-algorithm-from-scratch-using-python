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


def euclidien_dist_to_all(X,i):
        euclidien_dist_to_current_point = []
        for j in range(0,len(X)):
            euclidien_dist_to_current_point.append(euclidien_dist(X,i,j))
        euclidien_dist_to_current_point = np.array(euclidien_dist_to_current_point)
        return np.array(euclidien_dist_to_current_point)


def calc_close_pts_num(X,epsilon):
        res = [-1 for _ in range(len(X))]
        for i in range(len(res)):
            euclidien_dist_to_current_point = euclidien_dist_to_all(X,i)
            res[i] = len(euclidien_dist_to_current_point[euclidien_dist_to_current_point <= epsilon])
        return res


def assigne_clusters(clusters,epsilon,index,current_cluster_number):
    # step one : put the cluster of  the point to current cluster number
    clusters[index] = current_cluster_number
    # step two : identify indexes of all the close points
    close_points_indexes = []
    dist_to_all = euclidien_dist_to_all(X, index)
    for i in range(len(list(dist_to_all))):
        if dist_to_all[i] <= epsilon:
            close_points_indexes.append(i)
    for item in close_points_indexes:
        if clusters[item] == -1:
            assigne_clusters(clusters,epsilon,item, current_cluster_number)

def dbscan(X, epsilon, minPts):
    # first we need to calculate  how many points close to each point
    close_pts = calc_close_pts_num(X,epsilon)
    #print(f"close_pts : {close_pts}")
    # we will determine the core-points and non core-points using by comparing the  close points to each point and the minPts hyper param
    core_or_not_info = ['core' if i >= minPts else 'non-core' for i in close_pts]
    # now we will go to each core point and assigne it and it's neighbors to a cluster
    clusters = [-1 for _ in range(len(X))]
    # getting a random core point
    max_iter = 1000
    iter_ = 0
    random_core = rd.randint(0,len(X)-1)
    while core_or_not_info[random_core] != 'core':
        random_core = rd.randint(0, len(X) - 1)
        if iter_ == max_iter:
            print("each point creates a cluster")
            return
        iter_ += 1

    current_cluster_number = 0
    while True:
        assigne_clusters(clusters,epsilon,random_core,current_cluster_number)
        #print(f"tour : {current_cluster_number+1} ends with {clusters}")
        done = True
        for i in range(len(core_or_not_info)):
            if core_or_not_info[i] == 'core' and clusters[i] == -1:
                random_core = i
                done = False
        current_cluster_number += 1
        if done:
            break
    return clusters


def generate_x():
    import numpy as np
    import matplotlib.pyplot as plt

    num_points = 200

    theta1 = np.linspace(0, np.pi, num_points)
    theta2 = np.linspace(np.pi, 2 * np.pi, num_points)

    # generate a random set of radii between 0.5 and 1.5 for each half circle
    r1 = np.random.uniform(0.5, 0.8, num_points)
    r2 = np.random.uniform(0.5, 0.8, num_points)

    x1 = r1 * np.cos(theta1) - 1
    y1 = r1 * np.sin(theta1)

    x2 = r2 * np.cos(theta2) - 0.2
    y2 = r2 * np.sin(theta2) + 0.2

    half_circle1 = np.column_stack((x1, y1))
    half_circle2 = np.column_stack((x2, y2))

    X = np.concatenate((half_circle1, half_circle2))

    return X


if __name__ == "__main__":
    X = generate_x()

    plt.scatter(X[:, 0], X[:, 1])
    plt.title("initial points")
    plt.show()
    labels = dbscan(X, 0.1, 3)


    plt.scatter(X[:,0],X[:,1],c=labels)
    plt.title("DBSCAN result")
    plt.show()

    from sklearn.cluster import KMeans
    km =  KMeans(n_clusters=2)
    lbls = km.fit_predict(X)


    plt.scatter(X[:,0],X[:,1],c=lbls)
    plt.title("K-Means result")
    plt.show()

