"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
 
"""

from .alignment import mybio
from .alignment.blossum import distance_blosum_normalised_gap_to_zero, distance_blosum_normalised


from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import squareform

import numpy as np
from kneed import KneeLocator




def kmedoids_clustering(dist_matrix: np.array, max_nb_clusters: int) -> list:
    '''Clustering using the KMedoids algorithm. It will iterate up to max_nb_clusters number
    of clusters. It applied the elbow method on the loss ot find the optimal number of clusters. 

    Parameters
    ----------
        - dist_matrix: np.array
            A (N,N) matrix that represent the distance among and betweenN sequences.
        - max_nb_clusters: int
            Maximun number of clusters to try out.
    Returns
    -------
        - found_labels: list
            List of label for each element (in the same order) of dist_matrix.
                e.g., [1,1,1,3,1,4,4,2,1,2,2]

    '''
    if len(dist_matrix) < max_nb_clusters:
        max_nb_clusters = len(dist_matrix)

    sse = []
    for k in range(1,max_nb_clusters+1):
        c = KMedoids(n_clusters=k, metric='precomputed',
                         init='k-medoids++', method='pam',
                         random_state=0)
        c.fit(dist_matrix)
        sse.append(c.inertia_)  # within-cluster sum of distances

    kl = KneeLocator(range(1, max_nb_clusters+1), sse, curve="convex", direction="decreasing")
    best_nb_clust = kl.elbow
    if best_nb_clust is None:
        best_nb_clust = max_nb_clusters
        print(f'--> Could not find Elbos (max_cluster {max_nb_clusters} might be too low, so take max_cluster as optimal nb of clusters.')
    else: print(f'--> SSE elbow computed as {best_nb_clust} by kneed')


    final = KMedoids(n_clusters=int(best_nb_clust), metric='precomputed',
                     init='k-medoids++', method='pam',
                     random_state=0)
    final.fit(dist_matrix)
    founds_labels = final.labels_

    print(f'\n## Kmedoids clustering ##')

    kl = KneeLocator(range(1, max_nb_clusters+1), sse, curve="convex", direction="decreasing")
    print(f'--> SSE elbow computed as {kl.elbow} by kneed (out of {max_nb_clusters} tested)')

    return founds_labels
    


def get_k_medoids_clusters(aligned_sequences: list) -> list:
    ''' 
    Take aligned sequences as an input, compute BLOSUM matrix
    ditances, and cluster in using ELBOW method with k-medoids. 

    Returns a list of label clusters.
    '''
    # Compute distance matrices
    mat_for_dist = distance_blosum_normalised
    distance_mat = mybio.distance_matrix_from_aligned_sequences(aligned_sequences, matrix_for_distance=mat_for_dist)
    matrix_dissim = squareform(distance_mat)

    # Normalised nb of mutations with the average AHo length 
    norm_matrix_mut = np.zeros((distance_mat.shape))

    for i, seq_nb_mutsin in enumerate(distance_mat):
        for j, nb_mut in enumerate(seq_nb_mutsin):
            l = (len(aligned_sequences[i].replace('-',''))+len(aligned_sequences[j].replace('-','')))/2
            norm_matrix_mut[i,j] = nb_mut/l

    labels_kmedoids_blosum = kmedoids_clustering(norm_matrix_mut, max_nb_clusters=40)

    return labels_kmedoids_blosum