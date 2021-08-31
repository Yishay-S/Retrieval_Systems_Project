from interface import Regressor
from utils import get_data
from config import *

from statistics import mean
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import collections
import matplotlib.pyplot as plt
import random


class SimpleMean(Regressor):
    def __init__(self):
        self.user_ratings = {}
        self.user_means = {}

    def fit(self, X):
        # raise NotImplementedError
        for i, record in X.iterrows():
            # user_id = int(record['User_ID_Alias'])
            # movie_id = int(record['Movie_ID_Alias'])
            # rating = int(record['Ratings_Rating'])
            user_id = int(record[USER_COL_NAME_IN_DATAEST])
            item_id = int(record[ITEM_COL_NAME_IN_DATASET])
            rating = int(record[RATING_COL_NAME_IN_DATASET])
            if user_id not in self.user_ratings.keys(): self.user_ratings[user_id] = []
            self.user_ratings[user_id].append(rating)

        for user in self.user_ratings.keys():
            self.user_means[user] = mean(self.user_ratings[user])

    # def predict_on_pair(self, user: int, item: int):
    # raise NotImplementedError

    def calculate_rmse(self, val):
        error = 0
        for i, record in val.iterrows():
            # user_id = int(record['User_ID_Alias'])
            # movie_id = int(record['Movie_ID_Alias'])
            # rating = int(record['Ratings_Rating'])
            user_id = int(record[USER_COL_NAME_IN_DATAEST])
            item_id = int(record[ITEM_COL_NAME_IN_DATASET])
            rating = int(record[RATING_COL_NAME_IN_DATASET])
            error += pow(self.user_means[user_id] - rating, 2)
        return np.sqrt(error / len(val))


if __name__ == '__main__':
    train, validation = get_data()
    baseline_model = SimpleMean()
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
