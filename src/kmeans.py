import os
import csv
import time
import json
import wordcloud
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.cluster import KMeans
from tqdm import tqdm_notebook
from multiprocessing import Process
from multiprocessing import Pool, TimeoutError
from scipy import sparse
from src.utils.clustering_results import (
    create_metadata_file,
    create_groups_wordcloud,
    get_groups_data,
    create_groups_visualizations,
    plot_elems_per_group,
    calc_silhouette,
    plot_silhouette,
    save_clustered_df,
    plot_incidents_tests_statistics,
    save_metadata_incidents
)
sys.path.append('../')

warnings.filterwarnings('ignore')


def get_matrices_filenames(path):
    res = {}
    filenames = os.listdir(path)
    for filename in filenames:
        if '.csv' in filename or '.npz' in filename:
            res[filename.replace('.csv', '').replace('.npz', '')] = filename
    return res


def kmeans_cluster_and_save(
        df,
        df_original,
        K,
        filenames,
        clustering_info,
        index_col,
        groups_col,
        metric_col):

    start_time = time.time()

    #kmeans_res = None
    time_clust = None
# ---------------------------------------------------------------------------------------------------
    kmeans_res = KMeans(n_clusters=K).fit(df.values)
    time_clust = time.time() - start_time

    df_clustered = df.copy()
    df_clustered[groups_col] = kmeans_res.labels_
# ---------------------------------------------------------------------------------------------------

    silhouette_avg, sample_silhouette_values = calc_silhouette(
        df.values, kmeans_res.labels_, K)
    _, groups_silhouette_mean, groups_silhouette_std = plot_silhouette(kmeans_res.labels_, K,
                                                                       silhouette_avg,
                                                                       sample_silhouette_values,
                                                                       filenames['silhouettes'],
                                                                       plot_mean=False)

    groups_data, groups_count, durations_means, durations_stds, times = get_groups_count(K,
                                                                                         df_clustered.reset_index(),
                                                                                         df_original,
                                                                                         index_col,
                                                                                         groups_col,
                                                                                         metric_col,
                                                                                         path_groups_data=filenames[
                                                                                             'groups_data'],
                                                                                         path_sublogs=filenames[
                                                                                             'sublogs'])

    rows, columns = calc_subplots_grid(K)
    plot_groups_durations(
        times,
        K,
        filenames['durations'],
        rows=rows,
        columns=columns)

    # create_groups_visualizations(
    #     groups_data,
    #     path_barh=filenames['groups_bars'],
    #     path_feat_list=filenames['features_lists'])
    # print(">>> create_groups_visualizations")

    save_clustered_df(df_clustered, filenames['clustered_df'])
    print(">>> save_clustered_df")

    time_total = time.time() - start_time

    print(">>> Save metadata")
    save_metadata_incidents(
        filenames['metadata'],
        clustering_info,
        time_clust,
        time_total,
        groups_count,
        groups_silhouette_mean,
        groups_silhouette_std,
        durations_means,
        durations_stds)


def get_groups_count(
        k,
        clustered_df,
        original_df,
        index_col,
        groups_col,
        measure_col,
        path_groups_data=None,
        path_sublogs=None):
    elems = {}
    means = {}
    stds = {}
    times = {}
    groups_data = {}
    for i in range(0, k):
        group_data = clustered_df[clustered_df[groups_col] == i]
        groups_data[i] = group_data
        if path_groups_data:
            group_data[[groups_col]].to_csv(path_groups_data + str(i) + '.csv')
        group_indexes = list(group_data[index_col].unique())
        group_elem_original_data = original_df.loc[original_df[index_col].isin(
            group_indexes)]
        if path_sublogs:
            group_elem_original_data.to_csv(path_sublogs + str(i) + '.csv')
        elems[i] = len(group_elem_original_data)
        means[i] = group_elem_original_data[measure_col].mean()
        times[i] = group_elem_original_data[measure_col]
        stds[i] = group_elem_original_data[measure_col].std()
    return groups_data, elems, means, stds, times


def calc_subplots_grid(k):
    rows = 1
    columns = k
    if k == 6:
        rows = 2
        columns = 3
        return rows, columns
    if k == 7 or k == 8:
        rows = 2
        columns = 4
        return rows, columns
    if k == 9 or k == 10:
        rows = 2
        columns = 5
    return rows, columns


def plot_groups_durations(times, k, filename, rows=1, columns=None):

    if not columns:
        columns = k

    fig, axes = plt.subplots(
        nrows=rows, ncols=columns, figsize=(
            columns * 5, rows * 4))
    fig.subplots_adjust(hspace=.5, wspace=.5)
    axs = axes.ravel()

    for i in range(0, k):
        if times[i].any():
            data = list(
                times[i].reset_index().drop(
                    columns=['index'])['duration'])
            mean = np.mean(data)
            std = np.std(data)
            ax = axs[i]
            n, bins, patches = ax.hist(data)
            ax.axvline(mean, color='red', linestyle='--', linewidth=2)
            ax.fill_betweenx([0, np.max(n)],
                             [mean - std, mean - std],
                             [mean + std, mean + std],
                             color='r',
                             alpha=0.2,
                             hatch='/')
            ax.set_title('Group %s' % i)
            ax.set_xlabel('Time to incident resolution')
            ax.set_ylabel('Quantity of incidents')

    plt.savefig(filename, dpi=300)
    plt.close(fig)


def get_results_filenames(matrix, k):
    filenames = {}
    filenames['silhouettes'] = 'kmeans_results/silhouettes/%s_k%s.png' % (
        matrix, str(k))
    filenames['elems_per_group'] = 'kmeans_results/qtt_elements/%s_k%s.png' % (
        matrix, str(k))
    filenames['clustered_df'] = 'kmeans_results/clustered_dfs/%s_k%s' % (
        matrix, str(k))
    filenames['durations'] = 'kmeans_results/durations/%s_k%s.png' % (
        matrix, str(k))
    filenames['groups_wc'] = 'kmeans_results/wordclouds/%s_k%s_groups_wordcloud.png' % (
        matrix, str(k))
    filenames['groups_wcs'] = 'kmeans_results/wordclouds/%s_k%s_group' % (
        matrix, str(k))
    filenames['groups_bars'] = 'kmeans_results/bars/%s_k%s_group' % (
        matrix, str(k))
    # filenames['groups_treemap']='kmeans_results/treemaps/%s_k%s_group_' % (matrix,str(k))
    filenames['features_lists'] = 'kmeans_results/feat_lists/%s_k%s' % (
        matrix, str(k))
    filenames['groups_data'] = 'kmeans_results/groups_data/%s_k%s_group' % (
        matrix, str(k))
    filenames['sublogs'] = 'kmeans_results/sublogs/%s_k%s_group' % (
        matrix, str(k))
    filenames['metadata'] = metadata_filename
    return filenames


matrices_path = 'matrices/'
matrices = get_matrices_filenames(matrices_path)
matrices = {matrix: matrices[matrix]
            for matrix in matrices if 'csv' in matrices[matrix]}
matrices
{'individual_specialist_binary': 'individual_specialist_binary.csv',
 'individual_specialist_tf': 'individual_specialist_tf.csv',
 'individual_specialist_tfidf': 'individual_specialist_tfidf.csv'}

ks = [3, 5, 10]  # [2,3,5,7,10]

header = 'activity_represpipent,feature_selection,counting,dimensions,k,time_clust,time_total,'
header += 'group,elem_qtt,silhouette_mean,silhouette_std,duration_mean,duration_std'
metadata_filename = create_metadata_file('kmeans_results/metadata', header)

# Kmeans
#feature = column(activity / field) + '-' + value


def aplic_kmeans(log1):
    print(">>>> aplic_kmeans")
    for matrix_filename in tqdm_notebook(matrices):
        for k in ks:
            filenames = get_results_filenames(matrix_filename, k)
            log_vector = pd.read_csv(
                matrices_path +
                matrices[matrix_filename],
                index_col='number')
            matrix_str_id = matrix_filename.replace('.csv', '')
            dimensions = log_vector.shape[1]
            clustering_info = ','.join(matrix_str_id.split(
                '_')) + ',' + str(dimensions) + ',' + str(k)
            print(">>>> before kmeans_cluster_and_save")
            kmeans_cluster_and_save(
                log_vector,
                log1.copy(),
                k,
                filenames,
                clustering_info,
                'number',
                'group',
                'duration')
