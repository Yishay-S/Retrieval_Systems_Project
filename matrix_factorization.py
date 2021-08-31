from interface import Regressor
from typing import Dict
from utils import Config, get_data
import pandas as pd
from pandas import array
import numpy as np
from numpy import square, sqrt, array
from config import *

class MatrixFactorization(Regressor):
    def __init__(self, config):
        self.k = config.k
        self.gamma = config.gamma
        self.lr = config.lr
        self.total_epochs = config.epochs
        self.this_epoch = 0
        self.mu = None
        self.num_of_users = None
        self.num_of_items = None
        self.bu = None
        self.bi = None
        self.p = None
        self.q = None


    def record(self, covn_dict):
        epoch = "{:02d}".format(self.this_epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.4}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)


    def calc_regularization(self):
        return self.gamma*(np.sum(self.bu**2)+np.sum(self.bi**2)+np.sum(np.linalg.norm(self.q, axis=1)**2)+
                       np.sum(np.linalg.norm(self.p, axis=1)**2))


    def fit(self, X):
        #setting the arrays lengths for claculation components
        self.num_of_users =X[USER_COL_NAME_IN_DATAEST].max()+1
        self.num_of_items =X[ITEM_COL_NAME_IN_DATASET].max()+1
        self.bu = np.random.normal(0,0.01,self.num_of_users)
        self.bi = np.random.normal(0,0.01,self.num_of_items)
        self.p=np.random.normal(0,0.01, size=(self.num_of_users, self.k))
        self.q=np.random.normal(0,0.01, size=(self.num_of_items, self.k))
        self.mu = X[RATING_COL_NAME_IN_DATASET].mean() 
        # Running 10 Epcochs
        while self.this_epoch<self.total_epochs:
                    self.run_epoch(X)
                    train_MSE= pow(self.calculate_rmse(X),2)
                    train_target = train_MSE * X.shape[0] + self.calc_regularization()
                    con_epochs_score = {"target function (train)": train_target,"train_MSE": train_MSE}
                    print("train_mse:", train_MSE) 
                    self.record(con_epochs_score)
                    self.this_epoch += 1
        print("validation_rmse:")

    def run_epoch(self, data: np.array):
        for i, row in data.iterrows():
            # check the indexes are within the valid ranges for users and items
             if(row[USER_COL_NAME_IN_DATAEST] >=0 and row[USER_COL_NAME_IN_DATAEST] <self.num_of_users and row[ITEM_COL_NAME_IN_DATASET] >=0 and row[ITEM_COL_NAME_IN_DATASET] <self.num_of_items): 
                pu=self.p[row[USER_COL_NAME_IN_DATAEST]]# examine specific user from p 
                qi=self.q[row[ITEM_COL_NAME_IN_DATASET]]# examine specific item from q 
                r_hat=self.mu+self.bu[row[USER_COL_NAME_IN_DATAEST]]+self.bi[row[ITEM_COL_NAME_IN_DATASET]]+np.matmul(pu,np.transpose(qi)) # prediction
                error=row[RATING_COL_NAME_IN_DATASET]-r_hat # error is the delta between the real rating to our prediction
#               updating steps:
                self.bu[row[USER_COL_NAME_IN_DATAEST]]=self.bu[row[USER_COL_NAME_IN_DATAEST]]+self.lr*(error-self.gamma*self.bu[row[USER_COL_NAME_IN_DATAEST]]) # update user_biases
                self.bi[row[ITEM_COL_NAME_IN_DATASET]]=self.bi[row[ITEM_COL_NAME_IN_DATASET]]+self.lr*(error-self.gamma*self.bi[row[ITEM_COL_NAME_IN_DATASET]]) #update item_biases
                pu=pu+self.lr*(error*qi-self.gamma*pu) #update pu
                qi=qi+self.lr*(error*pu-self.gamma*qi) #update qi
                self.p[row[USER_COL_NAME_IN_DATAEST]]=pu #update the specific user row in p
                self.q[row[ITEM_COL_NAME_IN_DATASET]]=qi #update the specific item row in q         

    def predict_on_pair(self, user, item):
        x=self.mu+self.bu[user]+self.bi[item]+ np.matmul(self.p[user, :],np.transpose(self.q[item, :]))
        return x             

    def calculate_rmse(self, val: np.array):
        error=0
        for i,record in val.iterrows():
            if(record[USER_COL_NAME_IN_DATAEST]==0 or record[ITEM_COL_NAME_IN_DATASET]==0):
                 continue
            user = record[USER_COL_NAME_IN_DATAEST]
            item = record[ITEM_COL_NAME_IN_DATASET]
            rating= record[RATING_COL_NAME_IN_DATASET]
            error += square(rating - self.predict_on_pair(user, item))
        return sqrt(error/val.shape[0])

if __name__ == '__main__':
    baseline_config = Config(
        lr=0.01,
        gamma=0.001,
        k=24,
        epochs=10)

    train, validation = get_data()
    baseline_model = MatrixFactorization(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
