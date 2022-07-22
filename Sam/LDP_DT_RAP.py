import math
import time as ti
from datetime import date, time

from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score

from pure_ldp.frequency_oracles.rappor import rappor_client
from treeRAP import Tree
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import numpy as np
from pure_ldp.frequency_oracles import LHClient, LHServer, DEClient, DEServer, HEClient, HEServer, RAPPORClient, \
    RAPPORServer
import DataPreprocessor

database_names=['adult','mushroom','iris','vote','car','nursery','spect','weightliftingexercises','htru']
epsilon_values=[0.01,0.1,0.5,1,2,3,5]
depth = 4

# 'adult','mushroom','iris','vote','car','nursery','spect','weightliftingexercises','htru'
def hash_perturb(io):
    g = client_olh.privatise(io)
    return g

def encode(df, c):
    perturbed_df = pd.DataFrame()
    y = df.iloc[: , -1]
    for x in df.columns:
        tempColumn = df.loc[:, x].apply(lambda item: item * c)
        tempColumn = tempColumn + df.iloc[: , -1]
        perturbed_df[x] = tempColumn
    g = perturbed_df.iloc[:, :-1]
    g.insert(len(g.columns),'label',y)
    return g

def decode(df, cat, c):
    df = pd.DataFrame(df)
    df.insert(len(df.columns),'label',cat)
    perturbed_df = pd.DataFrame()
    for x in df.columns:
        tempColumn = df.loc[:, x] - df.iloc[: , -1]
        tempColumn = tempColumn.apply(lambda item: item / c)
        perturbed_df[x] = tempColumn
    return perturbed_df.iloc[:, :-1].astype('int')


def perturb(df, e):
    perturbed_df = pd.DataFrame()
    i = 0
    print(do)
    print(df)
    for x in df.columns:
        print(i)
        epsilon = e
        d = do[i]
        i+=1
        # print('d')
        # print(d)
        f = round(1/(0.5*math.exp(epsilon/2)+0.5), 2)

        server_olh.update_params(epsilon, d)
        client_olh.update_params(epsilon, d, hash_funcs=server_olh.get_hash_funcs())
        tempColumn = df.loc[:, x].apply(lambda item: hash_perturb(item + 1))
        perturbed_df[x] = tempColumn
    return perturbed_df


for xx in database_names:
    b = DataPreprocessor.DataPreprocessor()
    X, y = b.get_data(xx)
    X = X.astype('int')
    feat = list(X.columns)
    print(feat)
    do = []
    for x in X.columns:
        do.append(max(X[x]) + 1)
    X.insert(len(X.columns),'label',y)
    c = max(y) + 1
    do = [gg * c for gg in do]
    epsilon = 10
    d = 10
    f = round(1/(0.5*math.exp(epsilon/2)+0.5), 2)
    server_olh = RAPPORServer(f, 128, 8, d)
    client_olh = RAPPORClient(f, 128, server_olh.get_hash_funcs(), 8)

    server_olh2 = DEServer(epsilon=epsilon, d=d)
    classifierDataFrame = pd.DataFrame()
    for epsilon_value in epsilon_values:
        j = encode(X, c)
        v = perturb(j.iloc[:, :-1], epsilon_value)
        print('wat')
        print(v.iloc[1,1])
        balanced_accuracy = []
        accuracy = []
        times = []
        f1 = []
        prec = []
        recall = []
        for i in range(10):
            i+=1
            clf = Tree(attrNames=feat, depth=depth, ldpMechanismClient=client_olh,
                       ldpMechanismServer=server_olh, epsilon_value=epsilon_value,
                       domainSize=do, max=c)
            X_train, X_test, y_train, y_test = train_test_split(v, y, test_size=0.2)
            X_train1, X_test1, y_train1, y_test1 = train_test_split(X.iloc[:, :-1], y, test_size=0.2)
            # X_test = decode(X_test, y_test, c)
            clf.fit(X_train, y_train)
            start = ti.time()
            pre = clf.predict(X_test1)
            stop = ti.time()
            balanced_accuracy.append(balanced_accuracy_score(y_test1, pre))
            accuracy.append(accuracy_score(y_test1, pre))
            f1.append(f1_score(y_test1, pre, average='weighted'))
            prec.append(precision_score(y_test1, pre, average='weighted'))
            recall.append(recall_score(y_test1, pre, average='weighted'))
            times.append(stop-start)
        scores = {'score_time': times, 'test_accuracy': accuracy, 'test_balanced_accuracy': balanced_accuracy, 'test_f1_macro': f1, 'test_precision_macro': prec,
                  'test_recall_macro': recall}
        scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
        rowName = 'dtde' + '/' + xx + '/' + 'depth' + str(depth) + '/'
        scoresDataFrame = scoresDataFrame.add_prefix(rowName)
        classifierDataFrame[epsilon_value] = scoresDataFrame
    classifierDataFrame.to_csv("./Experiments/test_run_" +
                               date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')

