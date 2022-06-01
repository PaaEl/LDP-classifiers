from datetime import date

import pandas
from sklearn.metrics import confusion_matrix
import c45
from sklearn import metrics
import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

from pure_ldp.frequency_oracles import LHClient, LHServer
import DataPreprocessor

database_names=['adult','car','nursery']
epsilon_values=[0.01,0.1,0.5,1,5]

def hash_perturb(io):
    g = client_olh.privatise(io)
    server_olh.aggregate(g)
    return g

def hash_perturb_get0(io):
    return io[0]

def recategorize(io, e, d):
    server_olh2.aggregate(io)
    olh_estimates2 = np.array([0])
    for i in range(0, d):
        olh_estimates2 = np.append(olh_estimates2, 0)
    huu = np.array(olh_estimates2)
    for i in range(1, d + 1):
        olh_estimates2[i - 1] = round(server_olh2.estimate(i, True))
    huu2 = np.subtract(olh_estimates2, huu)
    winner = np.argwhere(huu2 == np.amax(huu2))
    return np.random.choice(winner.flatten())

def perturb(df, e):
    perturbed_df_hash = pd.DataFrame()
    perturbed_df = pd.DataFrame()
    perturbed_df_hash0 = pd.DataFrame()
    list_estimates = [[]]
    list_aggregates = [[]]
    for x in df.columns:
        epsilon = e
        d = max(df[x]) + 1
        # print(d)
        olh_estimates = []
        server_olh.update_params(epsilon, d)
        client_olh.update_params(epsilon, d)
        server_olh2.update_params(epsilon, d)
        # print('ol')
        # print(olh_estimates2)
        tempColumn = df.loc[:, x].apply(lambda item: hash_perturb(item + 1))
        perturbed_df_hash[x] = tempColumn
        tempColumn = perturbed_df_hash.loc[:, x].apply(lambda item: hash_perturb_get0(item))
        # tempColumn= pandas.cut(tempColumn,d)
        perturbed_df_hash0[x] = tempColumn
        # tempColumn = perturbed_df_hash.loc[:, x].apply(lambda item: recategorize(item, epsilon, d))
        # perturbed_df[x] = tempColumn
        # print('where')
        # print(server_olh.aggregated_data)
        list_aggregates.append(server_olh.aggregated_data)
        for i in range(0, d):
            olh_estimates.append(round(server_olh.estimate(i + 1)))
        list_estimates.append(olh_estimates)
    return perturbed_df_hash0

for x in database_names:
    b = DataPreprocessor.DataPreprocessor()
    X, y = b.get_data(x)
    X = X.astype('int')
    # y = y.astype('int')
    print(X)
    epsilon = 10
    d = 10
    client_olh = LHClient(epsilon=epsilon, d=d, use_olh=True)
    server_olh = LHServer(epsilon=epsilon, d=d, use_olh=True)
    server_olh2 = LHServer(epsilon=epsilon, d=d, use_olh=True)
    olh_estimates2 = np.array([0])
    clfC = c45.C45()
    scores = cross_validate(clfC, X, y,
                            scoring=['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])
    scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
    scoresDataFrame.drop(labels=['fit_time', 'score_time'], inplace=True)
    rowName = 'Decision tree' + '/' + x + '/'
    scoresDataFrame = scoresDataFrame.add_prefix(rowName)
    scoresDataFrame.to_csv("./Experiments/test_run_" +
                           date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')

    classifierDataFrame = pd.DataFrame()
    for epsilon_value in epsilon_values:
        v = perturb(X, epsilon_value)
        clfC = c45.C45()
        scores = cross_validate(clfC, v, y,
                                scoring=['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro',
                                         'recall_macro'])
        scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
        scoresDataFrame.drop(labels=['fit_time', 'score_time'], inplace=True)
        rowName = 'Decision tree' + '/' + x + '/'
        scoresDataFrame = scoresDataFrame.add_prefix(rowName)
        classifierDataFrame[epsilon_value] = scoresDataFrame
    classifierDataFrame.to_csv("./Experiments/test_run_" +
                               date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')




