from sklearn.cluster import DBSCAN
import numpy as np

# generate sample data
X = np.random.randn(100, 2)

# create DBSCAN object with specified parameters
dbscan = DBSCAN(eps=0.5, min_samples=5)

# fit the DBSCAN model to the data
dbscan.fit(X)

# retrieve the labels and core samples
labels = dbscan.labels_
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

# print the number of clusters and noise points
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print(f'Estimated number of clusters: {n_clusters_}')
print(f'Estimated number of noise points: {n_noise_}')

# plot the results
import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('DBSCAN clustering')
plt.show()
