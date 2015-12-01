from constants import *
import numpy as np
import csv
from sklearn import cluster
import math
import random as ra


# Import CSV file and convert to numpy array
def csv_to_np(filename):
    mat = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) > 0:
                mat.append([float(x) for x in row])
    mat = np.array(mat)
    return mat


# Use a variation Lloyd's algorithm to cluster libraries
# Part 1: K-Means clustering
def kmeans_assign(loc, n):
    km = cluster.KMeans(n_clusters=4)
    km.fit(np.array(loc[:, 1:2]))
    labels = km.labels_
    centers = [x[0] for x in km.cluster_centers_]
    return labels, centers

# Part 2: Balancing cluster sizes to be roughly the same
def balance(labels, loc, centers, n, passes):
    # Compute the average cluster size
    size = math.ceil(float(len(loc)) / n)
    for x in range(passes):
        for i in range(n):
            cent = list(centers)
            cent[i] = 9999
            tmp = np.array([loc[x] for x in range(len(loc)) if labels[x] == i])
            # If the number of data points is larger than the average size..
            while len(tmp) > size:
                # Find the centroid of the closest other cluster and change
                j = 0
                coord = tmp[j, 1]
                newlabel = np.argmin([abs(x - coord) for x in cent])
                index = [x for x in range(len(loc)) if coord == loc[x, 1]][0]
                labels[index] = newlabel
                tmp = np.array([loc[x]
                                for x in range(len(loc)) if labels[x] == i])
                j += 1
    return labels


# Get initial visit frequency based on demand
def dtof(d):
    f = [2] * len(d)
    for i in range(len(d)):
        if d[i] >= 75:
            f[i] = 3
        if d[i] >= 150:
            f[i] = 5
    return f


# Split all libraries into two days based on visit frequency
def day_split(loc, f):
    day1 = [loc[x] for x in range(len(loc)) if f[x] == 3 or f[x] == 5]
    day1 = np.array(day1)
    f1 = [x for x in f if f[x] == 3 or f[x] == 5]
    day2 = [loc[x] for x in range(len(loc)) if f[x] == 2 or f[x] == 5]
    day2 = np.array(day2)
    f2 = [x for x in f if f[x] == 2 or f[x] == 5]
    return day1, day2, f1, f2


# Split data based on clusters
def cluster_split(loc, day, l, n):
    dc = []
    for i in range(n):
        tmp = np.array([day[x] for x in range(len(day)) if l[x] == i])
        if 0 not in tmp[:, 0]:
            tmp = np.vstack((loc[0], tmp))
        dc.append(tmp)
    return dc


# Truncate the distance matrix
def dist_trunc(dist, dc):
    indices = dc[:, 0]
    indices = list(indices)
    trunc = dist[:, indices]
    trunc = trunc[indices, :]
    return trunc


# Get distance matrices for each cluster
def cluster_dist(dc, dist):
    df = []
    for i in range(len(dc)):
        df.append(dist_trunc(dist, dc[i]))
    return df


# Solve TSP for a given list of nodes
# Using the nearest insertion heuristic
def solve_tsp(df, dc):
    mat = np.copy(df)
    path = [0]
    indices = []
    mat[:, 0] = 99
    node = 0
    total = 0
    while len(path) != len(mat):
        nxt = np.argmin(mat[node])
        mincost = mat[node, nxt]
        insert = len(path)
        if len(indices) > 3:
            for i in range(0, len(indices) - 1):
                # Cost of inserting between two nodes
                cost = df[indices[i], nxt] + df[nxt, indices[i+1]] - \
                    df[indices[i], indices[i + 1]]
                if cost < mincost:
                    mincost = cost
                    insert = i + 1
        total += mincost
        total += 10
        # Insert at the location of minimum cost
        indices = indices[0:insert] + [nxt] + indices[insert:len(indices)]
        path = path[0:insert] + [dc[nxt, 0]] + path[insert:len(path)]
        mat[:, nxt] = 99
        node = nxt
    total += df[0, node]
    return path, total

# Wrapper for k-means related functions
def kmeans_wrapper(day1, day2, n):
    labels1, centers1 = kmeans_assign(day1, n)
    labels2, centers2 = kmeans_assign(day2, n)
    labels1 = balance(labels1, day1, centers1, n, 50)
    labels2 = balance(labels2, day2, centers2, n, 50)
    return labels1, labels2

# Wrapper for splitting data
def split_wrapper(loc, day1, labels1, day2, labels2, dist):
    dc1 = cluster_split(loc, day1, labels1, 4)
    dc2 = cluster_split(loc, day2, labels2, 4)
    df1 = cluster_dist(dc1, dist)
    df2 = cluster_dist(dc2, dist)
    return dc1, dc2, df1, df2


# Shuffle paths in the event that time > 160
def shuffle(dc, df, labels, day, dist, loc):
    times = []
    paths = []
    for i in range(len(df)):
        path, time = solve_tsp(df[i], dc[i])
        times.append(time)
        paths.append(path)
    l = [x for x in times if x > 160]
    ldc = [dc[x] for x in range(len(dc)) if times[x] > 160]
    count = 0
    while len(l) > 0:
        # Take a random row
        r = ra.randrange(1, len(ldc[0]))
        row = ldc[0][r]
        coord = row[1]
        # Change its label, recompute data chunks with paths & times
        index = [x for x in range(len(day)) if coord == day[x, 1]][0]
        labels[index] = (labels[index] + 1) % 4
        dc = cluster_split(loc, day, labels, 4)
        df = cluster_dist(dc, dist)
        times = []
        paths = []
        for i in range(len(df)):
            path, time = solve_tsp(df[i], dc[i])
            times.append(time)
            paths.append(path)
        l = [x for x in times if x > 160]
        ldc = [dc[x] for x in range(len(dc)) if times[x] > 160]
        count += 1
        # Break if this still doesn't work after 100 iterations
        if count > 100:
            break
    # If it doesn't work, set an arbitrarily high total time
    if count > 100:
        total_time = 100000
    else:
        total_time = sum(times)
    return dc, df, labels, times, paths, total_time


# Wrapper for tsp and shuffling
def tsp_shuffle_wrapper(dc1, dc2, df1, df2, labels1, labels2, day1, day2, dist, loc):
    dc1, df1, labels1, times1, paths1, total_time1 = shuffle(
        dc1, df1, labels1, day1, dist, loc)
    dc2, df2, labels2, times2, paths2, total_time2 = shuffle(
        dc2, df2, labels2, day2, dist, loc)
    times = times1 + times2
    paths = paths1 + paths2
    total_time = total_time1 + total_time2
    return times, paths, total_time
