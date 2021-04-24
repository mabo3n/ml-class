#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import cdist
from sys import stdin
from time import time

df_raw = pd.read_csv(stdin, sep=',')

df_raw.describe()
df_raw.head()

df_train = df_raw[df_raw.columns[:-1]]

# Compute distortions and wcss

def get_distance_from_nearest_center(points, cluster_centers):
    return np.min(
        cdist(points, cluster_centers, 'euclidean'),
        axis=1
    )

cluster_sizes = np.arange(1, 100)
distortions = []
wcss = []

for n in cluster_sizes:
    model = KMeans(n_clusters=n, random_state=0).fit(df_train)
    distortion = np.mean(
        get_distance_from_nearest_center(df_train, model.cluster_centers_)
    )
    distortions.append(distortion)
    wcss.append(model.inertia_)

# Compute distances from the line between the edges

def edges(x):
    return (x[0], x[-1])

def get_distance_from_line(line, point):
    ''' Source: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
    '''
    (x1, y1), (x2, y2) = line
    (x0, y0) = point
    return np.abs((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1)) \
        / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_distance_from_edges_line(points):
    line = edges(points)
    for point in points:
        yield get_distance_from_line(line, point)

distortion_points = list(zip(cluster_sizes, distortions))
distortion_distances = list(get_distance_from_edges_line(distortion_points))
distortion_elbow = cluster_sizes[distortion_distances.index(max(distortion_distances))]

wcss_points = list(zip(cluster_sizes, wcss))
wcss_distances = list(get_distance_from_edges_line(wcss_points))
wcss_elbow = cluster_sizes[wcss_distances.index(max(wcss_distances))]

distortion_distances_norm = minmax_scale(distortion_distances)
wcss_distances_norm = minmax_scale(wcss_distances)

# Plot

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 10))

ax1.plot(cluster_sizes, distortions, '.-r',
         label='avg distance to cluster center')
ax1.legend(loc='upper right')
ax1.grid()
ax1.plot(edges(cluster_sizes), edges(distortions), ':r', alpha=.2)

ax2.plot(cluster_sizes, wcss, '.-b',
         label='within-cluster sum of squares')
ax2.legend(loc='upper right')
ax2.grid()
ax2.plot(edges(cluster_sizes), edges(wcss), ':b', alpha=.2)

ax3.set_title('Normalized distance to edges\' line')
ax3.set_ylim((0, 1.1))
ax3.plot(cluster_sizes, distortion_distances_norm, '-r', alpha=.4)
ax3.plot(cluster_sizes, wcss_distances_norm, '-b', alpha=.4)
ax3.set_xticks(cluster_sizes)
plt.setp(ax3.get_xticklabels(), rotation=90)
ax3.grid()

ax3.set_xlabel('cluster size')

def highlight_highest_point(ax, x, y, *args):
    ymax = max(y)
    xmax = x[list(y).index(ymax)]
    ax.plot([xmax], [ymax], alpha=.4, *args)

highlight_highest_point(ax3, cluster_sizes, distortion_distances_norm, 'xr')
highlight_highest_point(ax3, cluster_sizes, wcss_distances_norm, 'xb')

plt.savefig(f'kmeans-{time()}.png')
