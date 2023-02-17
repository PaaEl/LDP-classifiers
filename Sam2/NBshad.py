import copy

import numpy as np

import DTTreeArt
import pandas as pd
from sklearn.model_selection import train_test_split


class Shad_tree():
    def create_shadow(self, x, y, nr, c, a, b):
        x_train = x[0:0]
        y_train = []
        x_test = x[0:0]
        y_test = []
        x_train_pred = []
        x_test_pred = []
        # print(x)
        # print(x_test_pred)

        for i in range(nr):
            print(i)
            ab, ac, ad, ae = train_test_split(a, b, test_size=0.8)
            target_train_size = len(x) // 2
            X_train1 = x[:target_train_size ]
            y_train1 = y[:target_train_size ]
            X_test1 = x[target_train_size:]
            y_test1 = y[target_train_size:]
            frames = [x_train, X_train1]
            x_train = pd.concat(frames)
            frames2 = [x_test, X_test1]
            x_test = pd.concat(frames2)
            print(type(y_train1))
            print(type(y_train))
            y_train = y_train + y_train1.tolist()
            y_test = y_test + y_test1.tolist()
            clf = c
            clf.fit( X_train1,y_train1)
            print('lukt')
            x_train_pred = x_train_pred + y_train1.tolist()
            pred = clf.predict(X_test1)
            x_test_pred = x_test_pred + pred

        return x_train, y_train, x_train_pred, x_test, y_test, x_test_pred
            # print(x_train)