import numpy as np

from interface import Regressor
from utils import get_data, Config
from config import *

import os
import csv

from interface import Regressor

from statistics import mean
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import collections
import matplotlib.pyplot as plt
import random

from tqdm import tqdm


class KnnItemSimilarity(Regressor):
    def __init__(self, config):
        self.K = config.k
        self.global_mean = None
        self.items_means = {}
        self.items_corr = {}
        self.pairs_items_corr = {}
        self.items_ratings = {}

    def fit(self, X):
        self.global_mean = X[RATING_COL_NAME_IN_DATASET].mean()
        for i, record in X.iterrows():
            user_id = int(record[USER_COL_NAME_IN_DATAEST])
            item_id = int(record[ITEM_COL_NAME_IN_DATASET])
            rating = int(record[RATING_COL_NAME_IN_DATASET])
            if item_id not in self.items_ratings.keys(): self.items_ratings[item_id] = {}
            self.items_ratings[item_id][user_id] = rating
            if item_id not in self.items_corr.keys(): self.items_corr[item_id] = {}

        if os.path.isfile(CORRELATION_PARAMS_FILE_PATH):
            self.upload_params()

        else:
            self.items_means = dict(
                X[[ITEM_COL_NAME_IN_DATASET, RATING_COL_NAME_IN_DATASET]].groupby(ITEM_COL_NAME_IN_DATASET,
                                                                                  as_index=False).mean().values)
            self.build_item_to_itm_corr_dict(X)
            self.save_params()

    def build_item_to_itm_corr_dict(self, data):
        for i in tqdm(self.items_ratings.keys()):
            if i not in self.items_corr.keys(): self.items_corr[i] = {}
            for j in self.items_ratings.keys():
                if i == j:
                    continue
                else:
                    mone = 0
                    mechane = 0
                    mechane_i = 0
                    mechane_j = 0
                    intersection_users = list(set((self.items_ratings[i]).keys()) & set((self.items_ratings[j]).keys()))
                    if len(intersection_users) > 0:
                        for user in intersection_users:
                            mone += (self.items_ratings[i][user] - self.items_means[i]) * (
                                        self.items_ratings[j][user] - self.items_means[j])
                            mechane_i += pow(self.items_ratings[i][user] - self.items_means[i], 2)
                            mechane_j += pow(self.items_ratings[j][user] - self.items_means[j], 2)
                        mechane = np.sqrt(mechane_i) * np.sqrt(mechane_j)
                        if mechane == 0:
                            continue
                        else:
                            if mone / mechane < 0:
                                continue
                            else:
                                self.items_corr[i][j] = mone / mechane
                                pair = frozenset({i, j})
                                if pair not in self.pairs_items_corr.keys(): self.pairs_items_corr[
                                    pair] = mone / mechane

    def predict_on_pair(self, user, item):
        if item not in self.items_ratings.keys(): return self.global_mean
        relevant_items = []
        for i in self.items_ratings.keys():
            if i == item:
                continue
            else:
                if user in self.items_ratings[i].keys() and i in self.items_corr[
                    item].keys() and i not in relevant_items: relevant_items.append(i)
        if len(relevant_items) == 0: return self.global_mean
        relevant_item_corr_dict = {x: self.items_corr[item][x] for x in relevant_items}
        mone = 0
        mechane = 0
        for j, sim in sorted(relevant_item_corr_dict.items(), key=lambda item: item[1], reverse=True)[:self.K]:
            mone += sim * self.items_ratings[j][user]
            mechane += sim
        return mone / mechane

    def upload_params(self):
        self.corr_df = pd.read_csv(CORRELATION_PARAMS_FILE_PATH)
        for i, record in self.corr_df.iterrows():
            item_1 = np.int16(record["item_1"])
            item_2 = np.int16(record["item_2"])
            sim = np.float32(record["sim"])
            if item_1 not in self.items_corr.keys(): self.items_corr[item_1] = {}
            if item_2 not in self.items_corr.keys(): self.items_corr[item_2] = {}
            self.items_corr[item_1][item_2] = sim
            self.items_corr[item_2][item_1] = sim

    def save_params(self):  # save corr_matrix to csv file
        with open(CORRELATION_PARAMS_FILE_PATH, 'w') as csv_file:
            write = csv.writer(csv_file, lineterminator='\n')
            write.writerow(CSV_COLUMN_NAMES)
            items_pairs = []
            for key, value in self.pairs_items_corr.items():
                pair = list(key)
                item_1 = pair[0]
                item_2 = pair[1]
                sim = value
                write.writerow([np.int16(item_1), np.int16(item_2), np.float32(sim)])

    def calculate_rmse(self, val):
        error = 0
        for i, record in val.iterrows():
            user_id = int(record[USER_COL_NAME_IN_DATAEST])
            item_id = int(record[ITEM_COL_NAME_IN_DATASET])
            rating = int(record[RATING_COL_NAME_IN_DATASET])
            error += pow(self.predict_on_pair(user_id, item_id) - rating, 2)
        return np.sqrt(error / len(val))


if __name__ == '__main__':
    knn_config = Config(k=25)
    train, validation = get_data()
    knn = KnnItemSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))
