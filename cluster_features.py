from typing import Dict, List, Union
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

class ClusterFeatures:
    # we can specify the argument datatype using "argument_name : type"
    #features: the embedding matrix created by bert parent.
    #algorithm: Which clustering algorithm to use.
    #pca_k: If you want the features to be ran through pca, this is the components number.
    def __init__(self, features: ndarray, random_state: int = 12345,):
        # Get the PCA components of the features if pca_k is not None
        #if pca_k:
        #    self.features = PCA(n_components=pca_k).fit_transform(features)
        #else:
        self.features = features
        self.random_state = random_state

    # Define and return the clustering model:
    # -> is used to indicate the return data type of the function in this case it is either a gaussian model or kmeans model
    def _get_model(self, k: int) -> Union[GaussianMixture, KMeans]:
        #k: amount of clusters.
        #print(self.random_state)
        return KMeans(n_clusters=k, random_state=self.random_state)

    # # Get the mean of each cluster
    # def _get_centroids(self, model: Union[GaussianMixture, KMeans]) -> np.ndarray:
    #     #model: Clustering model.
    #     #return: array of centroids.
    #     # For kmeans algorithm, the centroids is attribute .cluster_centers_
    #     return model.cluster_centers_

    # Find the closest features to the clusters centroids
    def __find_closest_args(self, centroids: np.ndarray) -> Dict:
        # Initialize the norm of the minimum distance between the feature and centroid to large value(infinity)
        centroid_min = 1e10
        cur_arg = -1
        # Dictionary of the centroid number and the closest feature to it
        args = {}
        # List to keep track of the closest features so as not to add a point more than once
        used_idx = []

        # enumerate add a counter to each array element
        for j, centroid in enumerate(centroids):

            for i, feature in enumerate(self.features):
                # Get the norm of the distance between the feature and the centroid
                value = np.linalg.norm(feature - centroid)

                if value < centroid_min and i not in used_idx:
                    cur_arg = i
                    centroid_min = value

            used_idx.append(cur_arg)
            args[j] = cur_arg
            centroid_min = 1e10
            cur_arg = -1

        return args

    # Get the loss across different number of clusters to choose the best number of clusters using elbow method
    def calculate_elbow(self, k_max: int) -> List[float]:
        # k_max: K_max to calculate elbow for.
        # return: The inertias up to k_max.

        inertias = []
        # Get the model, fit it then use.inertia_ to get the sum of squared distances of samples to their closest cluster center
        # inertias represents the loss values at different clusters numbers
        for k in range(1, min(k_max, len(self.features))):
            model = self._get_model(k).fit(self.features)

            inertias.append(model.inertia_)

        return inertias

    # Determine the optimal number of clusters using the elbow method
    def calculate_optimal_cluster(self, k_max: int) -> int:
        # k_max: The max k to search elbow for.
        # return: The optimal cluster size.

        delta_1 = []
        delta_2 = []

        max_strength = 0
        k = 1

        inertias = self.calculate_elbow(k_max)

        # Compute the first and second derivative approximately
        for i in range(len(inertias)):
            delta_1.append(inertias[i] - inertias[i - 1] if i > 0 else 0.0)
            delta_2.append(delta_1[i] - delta_1[i - 1] if i > 1 else 0.0)

        # Choose the k corresponding to the max 2nd derivative - 1st derivative difference
        # So the best when 2nd deivative is positive --> convex, and 1st derivative is negative --> decreasing function
        # Which marks the elbow
        for j in range(len(inertias)):
            strength = 0 if j <= 1 or j == len(inertias) - 1 else delta_2[j + 1] - delta_1[j + 1]

            if strength > max_strength:
                max_strength = strength
                k = j + 1
        return k

    # Return the chosen sentences
    def cluster(self, ratio: float = 0.1, num_sentences: int = None) -> List[int]:
        # ratio: Ratio to use for clustering.
        # num_sentences: Number of sentences. Overrides ratio.
        # return: Sentences index that qualify for summary.

        if num_sentences is not None:
            if num_sentences == 0:
                return []

            k = min(num_sentences, len(self.features))
        else:
            k = max(int(len(self.features) * ratio), 1)
        model = self._get_model(k).fit(self.features)
        # model = 0
        # for i in range(3):
        #     if i == 0:
        #         self.random_state = 12345
        #     elif i == 1:
        #         self.random_state = 0
        #     else:
        #         self.random_state = 420
        #     current_model = self._get_model(k).fit(self.features)
        #     print(current_model.inertia_)
        #     if i == 0:
        #         model = current_model
        #         inertia = current_model.inertia_
        #     else:
        #         if current_model.inertia_ < inertia:
        #             model = current_model
        #             inertia = current_model.inertia_




        #centroids = self._get_centroids(model)
        centroids = model.cluster_centers_
        cluster_args = self.__find_closest_args(centroids)

        # Sort the sentence indices to print them in order
        sorted_values = sorted(cluster_args.values())
        return sorted_values

    # Note __call__ function is the same as self.cluster but allow the use of object of the
    # class as a function
    def __call__(self, ratio: float = 0.1, num_sentences: int = None) -> List[int]:
        ############me :add num_sentences
        return self.cluster(ratio, num_sentences)


