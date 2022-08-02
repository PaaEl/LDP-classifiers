import time as ti
from datetime import date, time

from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score

import tree_pretty
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import numpy as np
import DataPreprocessor
from pure_ldp.frequency_oracles import DEClient, DEServer

epsilon = 1
d = 1
des = DEServer(epsilon=epsilon, d=d)
dec = DEClient(epsilon=epsilon, d=d)
tree_a = tree_pretty

ldp_mechanism = {'de': (dec, des, tree_a)}
database_names=['adult','mushroom','iris','vote','car','nursery','spect','weightliftingexercises','htru']
epsilon_values=[0.01,0.1,0.5,1,2,3,5]
depth = 2

# , 'olh': (dec, des, tree_a), 'hr': (dec, des, tree_a),
#                  'he': (dec, des, tree_a), 'oue': (dec, des, tree_a), 'rap': (dec, des, tree_a)
# 0.01,0.1,0.5,1,2,3,
# 'adult','mushroom','iris','vote','car','nursery','spect','weightliftingexercises','htru'
'''From pure ldp, perturbs the data'''
def hash_perturb(io):
    g = client.privatise(io)
    return g

'''Connects the feature value to the label value'''
def encode(df, c):
    perturbed_df = pd.DataFrame()
    # print(df)
    y = df.iloc[: , -1]
    for x in df.columns:
        tempColumn = df.loc[:, x].apply(lambda item: item * c)
        tempColumn = tempColumn + df.iloc[: , -1]
        perturbed_df[x] = tempColumn
    g = perturbed_df.iloc[:, :-1]
    g.insert(len(g.columns),'label',y)
    # print(g)
    return g

def perturb(df, e):
    perturbed_df = pd.DataFrame()
    for x in df.columns:
        epsilon = e
        d = max(df[x]) + 1
        server.update_params(epsilon, d)
        client.update_params(epsilon, d)
        tempColumn = df.loc[:, x].apply(lambda item: hash_perturb(item + 1))
        perturbed_df[x] = tempColumn
    return perturbed_df

for xxx in ldp_mechanism:
    a = ldp_mechanism[xxx]
    server = a[1]
    client= a[0]
    tree = a[2]
    for xx in database_names:
        # getting the data in order
        b = DataPreprocessor.DataPreprocessor()
        X, y = b.get_data(xx)
        X = X.astype('int')
        feat = list(X.columns)
        do = []
        for x in X.columns:
            do.append(max(X[x]) + 1)
        X.insert(len(X.columns), 'label', y)
        c = max(y) + 1
        # gets domainsize for each feature
        do = [gg * c for gg in do]
        classifierDataFrame = pd.DataFrame()
        for epsilon_value in epsilon_values:
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            j = encode(X, c)
            v = perturb(j.iloc[:, :-1], epsilon_value)
            # print(X)
            # print(v)
            balanced_accuracy = []
            accuracy = []
            times = []
            f1 = []
            prec = []
            recall = []
            # ten times and get the average
            for i in range(10):
                i += 1
                clf = tree.Tree(attrNames=feat, depth=depth, ldpMechanismClient=DEClient(epsilon=epsilon, d=d),
                            ldpMechanismServer=DEServer(epsilon=epsilon, d=d), epsilon_value=epsilon_value,
                            domainSize=do, max=c)
                # train on connected data
                # X_train, X_test, y_train, y_test = train_test_split(v, y, test_size=0.2)
                # to test on data that hasn't been connected
                X_train1, X_test1, y_train1, y_test1 = train_test_split(X.iloc[:, :-1], y, test_size=0.8)
                clf.fit(X,y, v)
                start = ti.time()
                pre = clf.predict(X_train1)
                stop = ti.time()
                # gathering the results
                balanced_accuracy.append(balanced_accuracy_score(y_train1, pre))
                accuracy.append(accuracy_score(y_train1, pre))
                f1.append(f1_score(y_train1, pre, average='weighted'))
                prec.append(precision_score(y_train1, pre, average='weighted'))
                recall.append(recall_score(y_train1, pre, average='weighted'))
                times.append(stop - start)
            scores = {'score_time': times, 'test_accuracy': accuracy, 'test_balanced_accuracy': balanced_accuracy,
                      'test_f1_macro': f1, 'test_precision_macro': prec,
                      'test_recall_macro': recall}
            scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
            rowName = 'dtde' + '/' + xx + '/' + 'depth' + str(depth) + '/'
            scoresDataFrame = scoresDataFrame.add_prefix(rowName)
            classifierDataFrame[epsilon_value] = scoresDataFrame
        classifierDataFrame.to_csv("./Experiments/test_run_" +
                                   date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')