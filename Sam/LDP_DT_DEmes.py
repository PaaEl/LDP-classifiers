import time as ti
from datetime import date, time

from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score

from treeinv import Tree
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import numpy as np
from pure_ldp.frequency_oracles import LHClient, LHServer, DEClient, DEServer
import DataPreprocessor

database_names=['adult','mushroom','iris','vote','car','nursery','spect','weightliftingexercises','htru']
epsilon_values=[5]
depth = 4
# 0.01,0.1,0.5,1,2,3,
# 'adult','mushroom','iris','vote','car','nursery','spect','weightliftingexercises','htru'

'''From pure ldp, perturbs the data'''
def hash_perturb(io):
    g = client_olh.privatise(io)
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
'''unused'''
def decode(df, cat, c):
    df = pd.DataFrame(df)
    df.insert(len(df.columns),'label',cat)
    perturbed_df = pd.DataFrame()
    for x in df.columns:
        tempColumn = df.loc[:, x] - df.iloc[: , -1]
        tempColumn = tempColumn.apply(lambda item: item / c)
        perturbed_df[x] = tempColumn
    return perturbed_df.iloc[:, :-1].astype('int')

'''Perturbs using hash_perturb'''
def perturb(df, e):
    perturbed_df = pd.DataFrame()
    for x in df.columns:
        epsilon = e
        d = max(df[x]) + 1
        server_olh.update_params(epsilon, d)
        client_olh.update_params(epsilon, d)
        tempColumn = df.loc[:, x].apply(lambda item: hash_perturb(item + 1))
        perturbed_df[x] = tempColumn
    return perturbed_df


for xx in database_names:
    # getting the data in order
    b = DataPreprocessor.DataPreprocessor()
    X, y = b.get_data(xx)
    X = X.astype('int')
    feat = list(X.columns)
    # print('uni1')
    # print(X['relationship'].value_counts())
    # print('uni')
    # print(X['relationship'].value_counts())
    # print('uni')
    # print(X['education'].value_counts())
    # print('uni')
    # print(X['age'].value_counts())
    # print('uni')
    # print(X['occupation'].value_counts())
    # print('uni')
    # print(X['hours-per-week'].value_counts())
    print(feat)
    # X = X.drop('education', axis=1)
    # X = X.drop('relationship', axis=1)
    do = []
    for x in X.columns:
        do.append(max(X[x]) + 1)
    X.insert(len(X.columns),'label',y)
    c = max(y) + 1
    # gets domainsize for each feature
    do = [gg * c for gg in do]
    # getting the LDP mechanism, parameters are reset when used
    epsilon = 10
    d = 10
    client_olh = DEClient(epsilon=epsilon, d=d)
    server_olh = DEServer(epsilon=epsilon, d=d)
    server_olh2 = DEServer(epsilon=epsilon, d=d)
    classifierDataFrame = pd.DataFrame()
    for epsilon_value in epsilon_values:
        j = encode(X, c)
        v = perturb(j.iloc[:, :-1], epsilon_value)
        balanced_accuracy = []
        # print('uni')
        # print(j['label'].value_counts())
        # # print('uni')
        # print(v['mean_prof'].value_counts())
        # print('uni')
        # print(v['std_dev'].value_counts())
        # print('uni')
        # print(v['kurtosis'].value_counts())
        # print('uni')
        # print(v['skewness'].value_counts())
        # print('uni')
        # print(v['mean_dm_snr'].value_counts())
        # print('uni')
        # print(v['std_dev_dm_snr'].value_counts())
        # print('uni')
        # print(v['kurosis_dm_snr'].value_counts())
        # print('uni')
        # print(v['skewness_dm_snr'].value_counts())
        # v = v.drop('education',axis=1)
        # v = v.drop('relationship',axis=1)
        # del do[1]
        # del do[3]
        # print('uni')
        # print(v['relationship'].value_counts())
        accuracy = []
        times = []
        f1 = []
        prec = []
        recall = []
        # ten times and get the average
        for i in range(10):
            i+=1
            clf = Tree(attrNames=feat, depth=depth, ldpMechanismClient=DEClient(epsilon=epsilon, d=d),
                       ldpMechanismServer=DEServer(epsilon=epsilon, d=d), epsilon_value=epsilon_value,
                       domainSize=do, max=c)
            # train on connected data
            X_train, X_test, y_train, y_test = train_test_split(v, y, test_size=0.2)
            # to test on data that hasn't been connected

            X_train1, X_test1, y_train1, y_test1 = train_test_split(X.iloc[:, :-1], y, test_size=0.8)
            # X_test = decode(X_test, y_test, c)
            clf.fit(X_train, y_train)
            start = ti.time()
            pre = clf.predict(X_test1)
            stop = ti.time()
            # gathering the results
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

