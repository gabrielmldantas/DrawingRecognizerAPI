import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split

def load_dataset_with_categories():
    max_dataset_size = 123000
    datasets = [
        np.load('resources/datasets/airplane.npy')[:max_dataset_size],
        np.load('resources/datasets/cat.npy')[:max_dataset_size],
        np.load('resources/datasets/cow.npy')[:max_dataset_size],
        np.load('resources/datasets/crab.npy')[:max_dataset_size],
        np.load('resources/datasets/laptop.npy')[:max_dataset_size],
        np.load('resources/datasets/toothbrush.npy')[:max_dataset_size]
    ]

    all_datasets = np.concatenate((*datasets,))
    all_categories = []
    for index, dataset in enumerate(datasets):
        for _ in dataset:
            all_categories.append(index)
    return {'dataset': all_datasets, 'categories': all_categories}

def split_data(datasets, categories):
    return train_test_split(datasets, categories, test_size=0.33)
