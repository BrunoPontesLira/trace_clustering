import os
import csv
import json
import pandas as pd


def read_matrices(path, index_col=None, skip=None):
    res = {}
    csvs = os.listdir(path)
    for csv in csvs:
        if '.csv' in csv:
            if not skip or skip not in csv:
                res[csv.replace('.csv', '')] = pd.read_csv(path + csv)
    return res


def read_dataset(dataset_path):
    return pd.read_csv(dataset_path, sep=';', index_col=None)


def get_datasets(path, filenames):
    datasets = []
    for dataset_filename in filenames:
        datasets.append(read_dataset(os.path.join(path, dataset_filename)))

    return datasets
