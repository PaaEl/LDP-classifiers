import copy

import numpy as np

import DTTreeArt
import pandas as pd
from sklearn.model_selection import train_test_split


class Shad_tree():
    def create_shadow(self,X, x, y, nr, c, a, b):
        x_train = x[0:0]
        X_train = X[0:0]
        y_train = []
        x_test = x[0:0]
        X_test = X[0:0]
        y_test = []
        x_train_pred = []
        x_test_pred = []
        # print(x)
        # print(x_test_pred)

        for i in range(1):
            print(i)
            ab, ac, ad, ae = train_test_split(a, b, test_size=0.8)
            target_train_size = len(x) // 2
            x_train1 = x[:target_train_size ]
            X_train1 = X[:target_train_size]
            y_train1 = y[:target_train_size ]
            X_test1 = X[target_train_size:]
            x_test1 = x[target_train_size:]
            y_test1 = y[target_train_size:]
            X_shadow = X[:target_train_size ]
            frames = [x_train, x_train1]
            framesX = [X_train, X_train1]
            print('frames')
            print(x_train)
            print(x_train1)
            x_train = pd.concat(frames)
            X_train = pd.concat(framesX)
            frames2 = [x_test, x_test1]
            frames2X = [X_test, X_test1]
            X_test = pd.concat(frames2X)
            x_test = pd.concat(frames2)
            print(type(y_train1))
            print(type(y_train))
            y_train = y_train + y_train1.tolist()
            y_test = y_test + y_test1.tolist()
            clf = c
            clf.fit(X_shadow, y_train1, x_train1)
            print('lukt')
            x_train_pred = x_train_pred + y_train1.tolist()
            pred = clf.predict(X_test1)
            x_test_pred = x_test_pred + pred
            # x_train = x[0:0]
            # y_train = []
            # x_test = x[0:0]
            # y_test = []
            # x_train_pred = []
            # x_test_pred = []
            print(y_test)
            print(x_test_pred)
            print(y_train)
            print(x_train_pred)

        return x_train, y_train, x_train_pred, X_test, y_test, x_test_pred, X_train, x_test
            # print(x_train)