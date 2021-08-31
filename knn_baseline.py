import numpy as np
import pandas as pd
import pickle

from interface import Regressor
from utils import get_data, Config
from config import *


class KnnBaseline(Regressor):
    def __init__(self, config):
        self.K = config.k
        self.global_mean = None
        #self.items_means = {} #self.items_mean_rating = {} 
        self.items_ratings = {}
        self.items_corr = {}
        self.bu = None
        self.bi = None
        
    def fit(self, X: np.array):
        self.users = set(X[USER_COL_NAME_IN_DATAEST].unique())
        self.items = set(X[ITEM_COL_NAME_IN_DATASET].unique())
        for i, record in X.iterrows():
            user_id = int(record[USER_COL_NAME_IN_DATAEST])
            item_id = int(record[ITEM_COL_NAME_IN_DATASET])
            rating = int(record[RATING_COL_NAME_IN_DATASET])
            if item_id not in self.items_ratings.keys() : self.items_ratings[item_id] = {}
            self.items_ratings[item_id][user_id] = rating
            if item_id not in self.items_corr.keys() : self.items_corr[item_id] = {}
                
        # uploading params from KNN and linear_regression_baseline models
        self.upload_params()

    def predict_on_pair(self, user: int, item: int):
        # check if user and item in the train set
        if user not in self.users or item not in self.items: return self.global_mean
        relevant_items=[]
        for i in self.items_ratings.keys():
            if i == item: 
                continue
            else:
                if user in self.items_ratings[i].keys() and i in self.items_corr[item].keys() and i not in relevant_items: relevant_items.append(i)
        # if there are no KNNs, retrun the linear_regression_baseline prediction
        if len(relevant_items) == 0: return self.global_mean + self.bu[user] + self.bi[item]
        # calculate Baseline KNN
        relevant_item_corr_dict = {x: self.items_corr[item][x] for x in relevant_items}
        mone=0
        mechane=0
        for j,sim in sorted(relevant_item_corr_dict.items(), key=lambda item: item[1], reverse=True)[:self.K]:
            buj = self.global_mean + self.bu[user] + self.bi[j]
            mone += sim * (self.items_ratings[j][user] - buj)
            mechane += sim
        bui = self.global_mean + self.bu[user] + self.bi[item]
        if mechane == 0: return self.global_mean + self.bu[user] + self.bi[item]
        return bui + (mone / mechane)
    
    def calculate_rmse(self, val):
        error = 0
        for i, record in val.iterrows():
            user = int(record[USER_COL_NAME_IN_DATAEST])
            item = int(record[ITEM_COL_NAME_IN_DATASET])
            rating = int(record[RATING_COL_NAME_IN_DATASET])
            error += (self.predict_on_pair(user, item) - rating)**2
        return np.sqrt(error/val.shape[0])

    def upload_params(self):
        # upload KNN params from csv file
        self.corr_df=pd.read_csv(CORRELATION_PARAMS_FILE_PATH)
        for i, record in self.corr_df.iterrows():
            item_1 = np.int16(record["item_1"])
            item_2 = np.int16(record["item_2"])
            sim = np.float32(record["sim"]) 
            if item_1 not in self.items_corr.keys() : self.items_corr[item_1] = {}
            if item_2 not in self.items_corr.keys() : self.items_corr[item_2] = {}
            self.items_corr[item_1][item_2] = sim 
            self.items_corr[item_2][item_1] = sim
        # upload linear_regression_baseline params from pickle file
        file = open(BASELINE_PARAMS_FILE_PATH, 'rb')
        data = pickle.load(file)
        file.close()
        self.global_mean = data[0]
        self.bu = data[1]
        self.bi = data[2]
        
if __name__ == '__main__':
    baseline_knn_config = Config(k=25)
    train, validation = get_data()
    knn_baseline = KnnBaseline(baseline_knn_config)
    knn_baseline.fit(train)
    print(knn_baseline.calculate_rmse(validation))
