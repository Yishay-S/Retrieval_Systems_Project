from typing import Dict
import numpy as np
import pickle

from interface import Regressor
from utils import Config, get_data
from config import *



class Baseline(Regressor):
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.train_epochs = config.epochs
        self.n_users = None
        self.n_items = None
        self.user_biases = None  # b_u (users) vector
        self.item_biases = None  # # b_i (items) vector
        self.current_epoch = 0
        self.global_bias = None
        self.global_mean = None

    def record(self, covn_dict: Dict):
        epoch = "{:02d}".format(self.current_epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.4}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def calc_regularization(self):
        return self.gamma * (np.sum(self.item_biases**2) + np.sum(self.user_biases**2))

    def fit(self, X):
        self.n_users = X[USER_COL_NAME_IN_DATAEST].nunique()+1
        self.n_items = X[ITEM_COL_NAME_IN_DATASET].nunique()+1
        self.users = set(X[USER_COL_NAME_IN_DATAEST].unique())
        self.items = set(X[ITEM_COL_NAME_IN_DATASET].unique())
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.global_mean = X[RATING_COL_NAME_IN_DATASET].mean()
        while self.current_epoch < self.train_epochs:
            self.run_epoch(X)
            train_mse = np.square(self.calculate_rmse(X))
            train_objective = train_mse * X.shape[0] + self.calc_regularization()
            epoch_convergence = {"train_objective": train_objective,
                                 "train_mse": train_mse}
            self.record(epoch_convergence)
            self.current_epoch += 1
        self.save_params()
        
        
    def run_epoch(self, data: np.array):
        for i, record in data.iterrows():
            user = record[USER_COL_NAME_IN_DATAEST]
            item = record[ITEM_COL_NAME_IN_DATASET]
            rating  = record[RATING_COL_NAME_IN_DATASET]
            prediction = self.predict_on_pair(user, item)
            erorr = rating - prediction
            self.user_biases[user] += self.lr * (erorr - self.gamma * self.user_biases[user])
            self.item_biases[item] += self.lr * (erorr - self.gamma * self.item_biases[item])

    def predict_on_pair(self, user: int, item: int):
        if user not in self.users or item not in self.items:
            return self.global_mean
        else:
            return self.global_mean + self.user_biases[user] + self.item_biases[item]    
     
    def calculate_rmse(self, val):
        error=0
        for i, record in val.iterrows():
            user = int(record[USER_COL_NAME_IN_DATAEST])
            item = int(record[ITEM_COL_NAME_IN_DATASET])
            rating = int(record[RATING_COL_NAME_IN_DATASET])
            error += (self.predict_on_pair(user, item) - rating)**2
        return np.sqrt(error/val.shape[0])

    def save_params(self):
        baseline_params = [self.global_mean, self.user_biases, self.item_biases]
        pickle.dump(baseline_params, open(BASELINE_PARAMS_FILE_PATH, "wb"))

if __name__ == '__main__':
    baseline_config = Config(
        lr=0.001,
        gamma=0.001,
        epochs=10)

    train, validation = get_data()
    baseline_model = Baseline(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
