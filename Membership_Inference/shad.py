
import pandas as pd

'''Creates shadow models and returns the data used in training and testing and the predictions'''
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

        # loop is obsolete
        for i in range(1):
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
            x_train = pd.concat(frames)
            X_train = pd.concat(framesX)
            frames2 = [x_test, x_test1]
            frames2X = [X_test, X_test1]
            X_test = pd.concat(frames2X)
            x_test = pd.concat(frames2)
            y_train = y_train + y_train1.tolist()
            y_test = y_test + y_test1.tolist()

            clf = c
            clf.fit(X_shadow, y_train1, x_train1)
            x_train_pred = x_train_pred + y_train1.tolist()
            pred = clf.predict(X_test1)
            x_test_pred = x_test_pred + pred

        return x_train, y_train, x_train_pred, X_test, y_test, x_test_pred, X_train, x_test