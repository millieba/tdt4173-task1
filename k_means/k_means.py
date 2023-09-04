import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    def __init__(self, clusters=2, iterations=100):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.clusters = clusters
        self.iterations = iterations

    def initialize_centroids_kmeans_plus_plus(self, X):
        # Initialize the first centroid randomly from the dataset
        centroids = [X[np.random.choice(X.shape[0])]]

        # Repeat until we have a centroid for each cluster
        while len(centroids) < self.clusters:
            # Compute the squared distances of each data point to the nearest existing centroid (squaring the distances weights the points further away more heavily)
            squared_distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X])
            
            # Choose a new centroid from data points with a probability proportional to squared distance
            prob_dist = squared_distances / sum(squared_distances)
            new_centroid_index = np.random.choice(X.shape[0], p=prob_dist)
            centroids.append(X[new_centroid_index])

        return np.array(centroids)
    
    def initialize_centroids_maxmin(self, X):
        # Initialize the first centroid randomly from the dataset
        centroids = [X[np.random.choice(X.shape[0])]]

        # Repeat until we have a centroid for each cluster
        while len(centroids) < self.clusters:
            # Compute the distances of each data point to the nearest existing centroid
            distances = np.array([min(np.linalg.norm(x - c) for c in centroids) for x in X])

            # Choose a new centroid from data points with the maximum minimum distance
            new_centroid_index = np.argmax(distances)
            centroids.append(X[new_centroid_index])

        return np.array(centroids)

    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        X = X.values
        self.centroids = self.initialize_centroids_maxmin(X) # Initialize centroids using algorithm similar to maxmin

        for _ in range(self.iterations):
            distances = cross_euclidean_distance(X, self.centroids) # Compute distances between all points and centroids
            cluster_assignments = np.argmin(distances, axis=1) # Assign each point to the closest centroid (argmin returns the index of the minimum value=closest distance)

            # Compute new centroids by taking the mean of all points in each cluster
            new_centroids = np.array([X[cluster_assignments == cluster_index].mean(axis=0) for cluster_index in range(self.clusters)])

            if np.all(new_centroids == self.centroids): # To avoid unnecessary iterations, stop if centroids don't change
                break

            self.centroids = new_centroids

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        distances = cross_euclidean_distance(X.values, self.centroids) # Compute the distance between each point and each cluster center
        cluster_assignments = np.argmin(distances, axis=1) # Find the closest centroid for each point
        return cluster_assignments
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids

    
# --- Some utility functions 
def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        # distortion += ((Xc - mu) ** 2).sum(axis=1)
        distortion += ((Xc - mu) ** 2).sum()  # TODO: Check why this is the only way I can get this to work
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))


def find_best_model(X, clusters, number_of_restarts):
    best_model = None
    best_distortion = float('inf')

    for _ in range(number_of_restarts):
        model = KMeans(clusters=clusters)
        model.fit(X)
        distortion = euclidean_distortion(X, model.predict(X))
        if distortion < best_distortion:
            best_distortion = distortion
            best_model = model
    
    return best_model

def check_performance(X, clusters, number_of_restarts, runs=10):
    good_runs = 0
    for _ in range(runs):
        for _ in range(number_of_restarts):
            best_model = find_best_model(X, clusters, number_of_restarts)
            model_1 = best_model
            distortion = euclidean_distortion(X, model_1.predict(X))
            silhouette = euclidean_silhouette(X, model_1.predict(X))
            if distortion < 4.0 and silhouette > 0.58: # Note: These values are specific to data_2
                good_runs += 1
                break
    print(f'Runs with good performance: {good_runs} / {runs} ({good_runs / (runs) * 100:.2f}%)')

    
if __name__ == '__main__':
    data_1 = pd.read_csv('k_means/data_2.csv')
    X = data_1[['x0', 'x1']]
    X = (X - X.min()) / (X.max() - X.min()) # Normalisation improves results for data_2, but slightly worsens results for data_1
    clusters = 8
    number_of_restarts = 3

    check_performance(X, clusters=clusters, number_of_restarts=number_of_restarts)
    best_model = find_best_model(X, clusters=clusters, number_of_restarts=number_of_restarts)
    model_1 = best_model
    
    z = model_1.predict(X)
    print(f'Silhouette Score: {euclidean_silhouette(X, z):.3f}')
    print(f'Distortion: {euclidean_distortion(X, z):.3f}')

    C = model_1.get_centroids()
    K = len(C)
    _, ax = plt.subplots(figsize=(5, 5), dpi=100)
    sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=X, ax=ax)
    sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
    ax.legend().remove()
    plt.show(block=False)
    plt.pause(3)
    plt.close()