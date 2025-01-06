
import os
import sys

import community
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from skfuzzy.cluster import cmeans
from sklearn import metrics
from sklearn.cluster import KMeans, MeanShift
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Industrial_time_series_analysis.Describe.describe_utils.mocar_util import Fermat


def cos_similar(arr1: np.ndarray, arr2: np.ndarray) -> float:

    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    len1 = len(farr1)
    len2 = len(farr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        farr2 = farr2[:len1]

    numer = np.sum(farr1 * farr2)
    denom = np.sqrt(np.sum(farr1 ** 2) * np.sum(farr2 ** 2))
    similar = numer / denom
    return (similar + 1) / 2


def similarityMatrix(Data_norm, Nsample, lwin, node_num):

    # Calculate covariance matrix by moving window
    cov_t = []
    for i in range(0, Nsample, lwin):
        Data_norm_t = pd.DataFrame(Data_norm[i: i + lwin, :])
        cov_t.append(Data_norm_t.corr())

    dMatrix = np.zeros((node_num, node_num))
    for i in range(0, node_num - 1):
        for j in range(i + 1, node_num):
            dMatrix[i, j] = cos_similar(cov_t[i].values, cov_t[j].values)

    Similarity = dMatrix + dMatrix.T + np.eye(node_num)
    return Similarity


def threhold_select(similarity_matrix):
    Threshold = []
    for i in range(0, len(similarity_matrix)):
        aa = np.unique(similarity_matrix[i, :])

        aa_interval = np.diff(aa)


        th_i = aa[np.argmax(aa_interval) + 1]
        Threshold.append(th_i)

    return min(Threshold)


def complexNetwork(similarity_matrix, threshold):

    similarity_up = np.triu(similarity_matrix, k=1)


    adjacency_matrix = similarity_matrix - threshold + np.diag([threshold] * similarity_matrix.shape[0])

    adjacency_matrix[adjacency_matrix > 0] = 1
    adjacency_matrix[adjacency_matrix < 0] = 0

    G = nx.from_numpy_array(adjacency_matrix)

    return adjacency_matrix, G


def mode_num_update(values):

    new_values = list(set(values))
    new_values.sort(key=values.index)

    mode_num = len(new_values)  # 模态数
    indexs = []

    for i in new_values:
        values = [1 - mode_num + new_values.index(i) if j == i else j for j in values]
        indexs.append('mode ' + str(new_values.index(i) + 1))
    labels = [i + mode_num for i in values]

    return mode_num, labels, indexs



def community_detection(adjacency_matrix, G):
    part = community.best_partition(G)
    values = [part.get(node) for node in G.nodes()]
    # 更新模态序号
    mode_num, labels, indexs = mode_num_update(values)
    metrics = pd.DataFrame(index=indexs, columns=['n', 'm'])
    sample_num_mode = []


    for i in indexs:
        Modei, metrics.loc[i, 'n'], metrics.loc[i, 'm'] = network_metrics(
            indexs.index(i) + 1, labels, adjacency_matrix)
        sample_num_mode.append(Modei)

    return labels, mode_num, sample_num_mode, metrics, indexs

def network_metrics(xmode, labels, adjacency_matrix):
    Modei = [i for i, x in enumerate(labels) if x == xmode]

    adjacencyi = adjacency_matrix[Modei].T[Modei]
    Gi = nx.from_numpy_array(adjacencyi)

    Gi_edges_num = Gi.size()



    return Modei, Modei.__len__(), Gi_edges_num



def fermat_dist_matrix(data_scaler):

    distances = distance_matrix(data_scaler, data_scaler)

    alpha = 3
    k = 100
    landmarks = 30

    f_exact = Fermat(alpha=alpha, path_method='FW')  # Initialize the model

    f_exact.fit(np.matrix(distances))  # Fit
    fermat_dist_exact = f_exact.get_distances()

    return fermat_dist_exact


def nearPSD(A, epsilon=0):
    A = np.array(A)
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1 / (np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    out = B * B.T
    return out


def generate_data_mode(mean, cov, real_mean, sample_num, seed_num=1):
    np.random.seed(seed_num)

    data_mode1 = np.random.multivariate_normal(mean, nearPSD(cov[0]), (sample_num[0],), 'raise') + real_mean[0]
    data_mode2 = np.random.multivariate_normal(mean, nearPSD(cov[1]), (sample_num[1],), 'raise') + real_mean[1]
    data_mode3 = np.random.multivariate_normal(mean, nearPSD(cov[2]), (sample_num[2],), 'raise') + real_mean[2]

    data = np.concatenate((data_mode1, data_mode2, data_mode3), axis=0)
    scaler = StandardScaler().fit(data)

    data_scaler = scaler.transform(data)
    return data, data_scaler, scaler


def performance_comparision(data_scaler, n_cluster, labels):
    performance = pd.DataFrame(index=['K-Means', 'FCM', 'Mean_shift', 'Proposed'], columns=['SC', 'CHI', 'DBI'])

    model_kmeans = KMeans(n_clusters=n_cluster)
    model_kmeans.fit(data_scaler)
    labels_kmeans = model_kmeans.labels_

    model_MS = MeanShift()
    model_MS.fit(data_scaler)
    labels_MS = model_MS.labels_

    center, u, u0, d, jm, p, fpc = cmeans(data_scaler.T, m=2, c=n_cluster, error=0.005, maxiter=1000)
    for i in u:
        labels_FCM = np.argmax(u, axis=0)


    _, labels_kmeans, _ = mode_num_update(labels_kmeans.tolist())
    _, labels_MS, _ = mode_num_update(labels_MS.tolist())
    _, labels_FCM, _ = mode_num_update(labels_FCM.tolist())

    performance.loc['K-Means', 'SC'] = metrics.silhouette_score(data_scaler, labels_kmeans, metric='euclidean')
    performance.loc['FCM', 'SC'] = metrics.silhouette_score(data_scaler, labels_FCM, metric='euclidean')
    performance.loc['Mean_shift', 'SC'] = metrics.silhouette_score(data_scaler, labels_MS, metric='euclidean')
    performance.loc['Proposed', 'SC'] = metrics.silhouette_score(data_scaler, labels, metric='euclidean')

    performance.loc['K-Means', 'CHI'] = metrics.calinski_harabasz_score(data_scaler, labels_kmeans)
    performance.loc['FCM', 'CHI'] = metrics.calinski_harabasz_score(data_scaler, labels_FCM)
    performance.loc['Mean_shift', 'CHI'] = metrics.calinski_harabasz_score(data_scaler, labels_MS)
    performance.loc['Proposed', 'CHI'] = metrics.calinski_harabasz_score(data_scaler, labels)

    performance.loc['K-Means', 'DBI'] = metrics.davies_bouldin_score(data_scaler, labels_kmeans)
    performance.loc['FCM', 'DBI'] = metrics.davies_bouldin_score(data_scaler, labels_FCM)
    performance.loc['Mean_shift', 'DBI'] = metrics.davies_bouldin_score(data_scaler, labels_MS)
    performance.loc['Proposed', 'DBI'] = metrics.davies_bouldin_score(data_scaler, labels)

    return performance, labels_kmeans, labels_MS, labels_FCM


def Condition_ecognition(df,A,model):
    X = pd.DataFrame(df).values
    y_pred = model.predict(X)
    B = y_pred
    segment_ends = [i + 1 for i, (current, next_val) in enumerate(zip(A, A[1:] + [None])) if current != next_val]
    segment_starts = [0] + segment_ends[:-1]
    most_repeated_in_B_segments = []
    for start, end in zip(segment_starts, segment_ends):
        segment_of_B = B[start:end]
        element_counts = {}
        for elem in segment_of_B:
            if elem in element_counts:
                element_counts[elem] += 1
            else:
                element_counts[elem] = 1
        most_repeated_elem = next((k for k, v in element_counts.items() if v == max(element_counts.values())), None)
        most_repeated_in_B_segments.append(most_repeated_elem)

    return most_repeated_in_B_segments, y_pred

def community_new(sample_abnormal, sample_mode_i, labels, adjacency_matrix, G, target_mode_num):
    indexs_new = []
    # indexs_new.append('Mode 0')
    for i in range(len(sample_abnormal)):
        for j in sample_abnormal[i]:
            labels[j] = 0

    for i in range(len(sample_mode_i)):
        indexs_new.append('Mode' + str(i + 1))
        for j in sample_mode_i[i]:
            labels[j] = i + 1

    indices_of_zero = [i for i, x in enumerate(labels) if x == 0]


    for index in indices_of_zero:

        if index > 0 and index < len(labels) - 1:
            prev_value = labels[index - 1]
            next_value = labels[index + 1]


            if prev_value == next_value:
                labels[index] = prev_value
            else:

                labels[index] = prev_value


    partition = {index: value for index, value in enumerate(labels)}
    current_count = len(set(partition.values()))


    while current_count != target_mode_num:
        if current_count > target_mode_num:
            labels = adjust_communities(G, labels, target_mode_num)
            current_count -= 1
        else:
            print("请重新设置边界条件")
            current_count += 1


    metrics_new = pd.DataFrame(index=indexs_new, columns=['n', 'm'])
    sample_num_mode_new = []
    # print(indexs_new)

    for i in indexs_new:
        # print(i)
        Modei_new, metrics_new.loc[i, 'n'], metrics_new.loc[i, 'm'] = network_metrics(
            indexs_new.index(i) + 1, labels, adjacency_matrix)
        sample_num_mode_new.append(Modei_new)

    return labels, len(indexs_new), sample_num_mode_new, metrics_new, indexs_new


def adjust_communities(G, labels, target_count):
    partition = {index: value for index, value in enumerate(labels)}
    current_count = len(set(partition.values()))


    inter_community_edges = defaultdict(int)


    for u, v in G.edges():
        if partition[u] != partition[v]:
            inter_community_edges[(partition[u], partition[v])] += 1


    max_edge = max(inter_community_edges.items(), key=lambda item: item[1])

    m_community, a_community = max_edge[0]
    for node in G.nodes():
        if partition[node] == a_community:
            partition[node] = m_community

    labels = [value for key, value in partition.items()]

    return labels