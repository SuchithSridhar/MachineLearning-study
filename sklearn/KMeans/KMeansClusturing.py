#!/usr/bin/python3

import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# First Unsupervised algorithm
# KMeans clustering algorithm


x, y = load_digits(return_X_y=True)
data = scale(x)

# Scale function shifts to -1 - 1
# This is to make processing easier



# K = len(np.unique(y))
k = 10
# Since 10 digits

samples, features = data.shape

# copied function from sklean website
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

clf = KMeans(
        n_clusters=k,
        init="random", # starting locations
        n_init=10 # times to start with different seed)
    )

bench_k_means(clf, "1", data)