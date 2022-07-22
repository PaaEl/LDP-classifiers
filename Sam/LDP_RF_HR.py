import time as ti
from datetime import date, time

from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score

from treeRFHR import Tree
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import numpy as np
from pure_ldp.frequency_oracles import LHClient, LHServer, DEClient, DEServer, HadamardResponseServer, \
    HadamardResponseClient
import DataPreprocessor

database_names=['adult','mushroom','iris','vote','car','nursery','spect','weightliftingexercises','htru']
epsilon_values=[0.01,0.1,0.5,1,2,3,5]
depth = 4
forest_size = 20
lis = []
lis_pred = []
frac_data = 0.5
# 0.01,0.1,0.5,1,2,3,5,10
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
    for x in df.columns:
        epsilon = e
        d = max(df[x]) + 1
        server_olh.update_params(epsilon, d)
        # print('hash')
        # print(server_olh.get_hash_funcs())
        client_olh.update_params(epsilon, d, hash_funcs=server_olh.get_hash_funcs())
        tempColumn = df.loc[:, x].apply(lambda item: hash_perturb(item + 1))
        perturbed_df[x] = tempColumn
    return perturbed_df

for xx in database_names:
    lis = []
    lis_pred = []
    b = DataPreprocessor.DataPreprocessor()
    X, y = b.get_data(xx)
    print(X)
    X = X.astype('int')
    feat = list(X.columns)
    # print(feat)
    do = []
    for x in X.columns:
        do.append(max(X[x]) + 1)
    X.insert(len(X.columns),'label',y)
    c = max(y) + 1
    do = [gg * c for gg in do]
    epsilon = 10
    d = 10
    server_olh = HadamardResponseServer(epsilon=epsilon, d=d)
    client_olh = HadamardResponseClient(epsilon, d, server_olh.get_hash_funcs())
    classifierDataFrame = pd.DataFrame()
    for epsilon_value in epsilon_values:
        lis_pred = []
        lis = []
        lislis = []
        T = X
        j = encode(X, c)
        v = perturb(j.iloc[:, :-1], epsilon_value)
        v.insert(len(v.columns), 'label', y)
        # print('v')
        # print(v)
        balanced_accuracy = []
        accuracy = []
        times = []
        f1 = []
        prec = []
        recall = []
        for i in range(forest_size):
            i += 1
            # print('v')
            # print(v)

            J = v.sample(frac=frac_data, axis='rows')
            J.iloc[:, :-1]
            clf = Tree(attrNames=feat, depth=depth, ldpMechanismClient=client_olh,
                       ldpMechanismServer=server_olh, epsilon_value=epsilon_value,
                       domainSize=do, max=c)
            X_train, X_test, y_train, y_test = train_test_split(J.iloc[:, :-1], J.iloc[:, -1:], test_size=0.2)
            # X_test = decode(X_test, y_test, c)
            clf.fit(X_train, y_train)
            lis.append(clf)
        X_train1, X_test1, y_train1, y_test1 = train_test_split(T.iloc[:, :-1], y, test_size=0.2)
        for l in lis:
            # print('x1')
            # print(X_test1)
            start = ti.time()
            pre = l.predict(X_test1)
            stop = ti.time()
            times.append(stop - start)
            lis_pred.append(pre)
            # print('lispred')
            # print(lis_pred)
            # print(len(lis_pred))
        for i in range(len(lis_pred[0])):

            res = [j[i] for j in lis_pred]
            lislis.append(res)
            i += 1
        # print(lislis)
        jj = [max(l, key=l.count) for l in lislis]
        # print(jj)
        # print(y_test1)
        # print(res)
        # print(y_test1)
        balanced_accuracy.append(balanced_accuracy_score(y_test1, jj))
        accuracy.append(accuracy_score(y_test1, jj))
        f1.append(f1_score(y_test1, jj, average='weighted'))
        prec.append(precision_score(y_test1, jj, average='weighted'))
        recall.append(recall_score(y_test1, jj, average='weighted'))
        # print('fdfs')
        # print(balanced_accuracy)
        # print(times)
        scores = {'score_time': sum(times), 'test_accuracy': accuracy, 'test_balanced_accuracy': balanced_accuracy,
              'test_f1_macro': f1, 'test_precision_macro': prec,
              'test_recall_macro': recall}
        scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
        rowName = 'rfde' + '/' + xx + '/' + 'depth' + str(depth) + '/' + 'size' + str(forest_size) + '/'
        scoresDataFrame = scoresDataFrame.add_prefix(rowName)
        classifierDataFrame[epsilon_value] = scoresDataFrame
    print(classifierDataFrame)
    classifierDataFrame.to_csv("./Experiments/test_run_" +
                           date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')


    #     for i in range(10):
    #         i+=1
    #         clf = Tree(attrNames=feat, depth=depth, ldpMechanismClient=DEClient(epsilon=epsilon, d=d),
    #                    ldpMechanismServer=DEServer(epsilon=epsilon, d=d), epsilon_value=epsilon_value,
    #                    domainSize=do, max=c)
    #         X_train, X_test, y_train, y_test = train_test_split(v, y, test_size=0.2)
    #         # X_train1, X_test1, y_train1, y_test1 = train_test_split(X.iloc[:, :-1], y, test_size=0.2)
    #         X_test = decode(X_test, y_test, c)
    #         clf.fit(X_train, y_train)
    #         start = ti.time()
    #         pre = clf.predict(X_test)
    #         stop = ti.time()
    #         balanced_accuracy.append(balanced_accuracy_score(y_test, pre))
    #         accuracy.append(accuracy_score(y_test, pre))
    #         f1.append(f1_score(y_test, pre, average='micro'))
    #         prec.append(precision_score(y_test, pre, average='micro'))
    #         recall.append(recall_score(y_test, pre, average='micro'))
    #         times.append(stop-start)
    #     scores = {'score_time': times, 'test_accuracy': accuracy, 'test_balanced_accuracy': balanced_accuracy, 'test_f1_macro': f1, 'test_precision_macro': prec,
    #               'test_recall_macro': recall}
    #     scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
    #     rowName = 'dtde' + '/' + xx + '/' + 'depth' + str(depth) + '/'
    #     scoresDataFrame = scoresDataFrame.add_prefix(rowName)
    #     classifierDataFrame[epsilon_value] = scoresDataFrame
    # classifierDataFrame.to_csv("./Experiments/test_run_" +
    #                            date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')