import math
from datetime import date

import pandas
from sklearn.metrics import confusion_matrix
import c45
from sklearn import metrics
import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

from pure_ldp.frequency_oracles import LHClient, LHServer, DEClient, DEServer
import DataPreprocessor

database_names=['mushroom','car','spect','weightliftingexercises','htru']
epsilon_values=[0.01,1,5]
depth = 10


def hash_perturb(io):
    g = client_olh.privatise(io)
    return g

# def gain(df, lis):

    # return ranked

def encode(df, c):
    perturbed_df = pd.DataFrame()
    for x in df.columns:
        d = max(df[x]) + 1
        # print('d')
        # print(d)
        tempColumn = df.loc[:, x].apply(lambda item: item * c)
        # print(tempColumn)
        # print(df.iloc[: , :-1])
        tempColumn = tempColumn + df.iloc[: , -1]
        # print(tempColumn)
        perturbed_df[x] = tempColumn
    return perturbed_df

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

def estimate(df, e):
    lis =[]
    for x in df.columns:
        epsilon = e
        d = max(df[x]) + 1
        server_olh.update_params(epsilon, d)
        client_olh.update_params(epsilon, d)
        df.loc[:, x].apply(lambda g: server_olh.aggregate(g))
        li = []
        for i in range(0, d):
            li.append(round(server_olh.estimate(i + 1)))
        lis.append(li)
    return lis

def not_neg(lis):
    t = [[abs(j) for j in y] for y in lis]
    return t

def rank(df, lis,  c):
    ran = []
    i = 0
    for x in df.columns:
        # print(df.info)
        # print(len(lis))
        # print('lisi')
        # print(lis[i])
        s = gain(lis[i],c)
        tu = (s, x)
        ran.append(tu)
        i+=1
    return ran
def entr(x):
    if x == 0:
        return 0
    else:
        return x * math.log2(x)
def gain(lis, c):
    fraction =[]
    prob = []
    b = sum(lis)
    # print(b)
    i =1
    while i < len(lis):
        # print('i')
        # print(i)
        bb = sum(lis [i-1:i+c-1])
        # print(bb)
        fraction.append(bb/b)
        j = i -1
        while j < (i+ c-1):
            bbb = lis[j]/bb
            prob.append(bbb)
            j+=1
        i +=c
    # print('frac')
    # print(fraction)
    # print(prob)
    enj = 0
    j = 0
    i = 1
    while i - 1 < len(fraction):
        # print('j')
        # print(j)
        # print(i * c)
        en = 0
        while j < (i * c):
            en += entr(prob[j])
            # print(prob[j])
            # print(entr(prob[j]))
            j += 1
            # print('en')
            # print(en)
        enj += fraction[i - 1] * en
        i += 1
    # print('enj')
    # print(enj)
    return 1 + enj


for x in database_names:
    b = DataPreprocessor.DataPreprocessor()
    X, y = b.get_data(x)
    X = X.astype('int')
    X.insert(len(X.columns),'label',y)
    # ffg = X.iloc[:, 0].value_counts()
    # print(ffg)
    # gdsd = X.loc[X.iloc[:, 0] == 3]
    # print(gdsd)
    # gdsda = gdsd.loc[X.iloc[:, -1] == 1]
    # print(gdsda)
    c = max(y) + 1
    imp =[]
    for i in range(c):
        imp.append(np.count_nonzero(y == i))
        print(imp)
    print('c')
    print(c)
    print(X)
    print(X.iloc[: , :-1])
    epsilon = 10
    d = 10
    client_olh = DEClient(epsilon=epsilon, d=d)
    server_olh = DEServer(epsilon=epsilon, d=d)
    server_olh2 = DEServer(epsilon=epsilon, d=d)
    olh_estimates2 = np.array([0])
    clfC = c45.C45()
    scores = cross_validate(clfC, X.iloc[: , :-1], y,
                            scoring=['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro',
                                     'recall_macro'])
    scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
    scoresDataFrame.drop(labels=['fit_time'], inplace=True)
    rowName = 'dtde' + '/' + x + '/'
    scoresDataFrame = scoresDataFrame.add_prefix(rowName)
    scoresDataFrame.to_csv("./Experiments/test_run_" +
                           date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')

    classifierDataFrame = pd.DataFrame()
    for epsilon_value in epsilon_values:
        j = encode(X,c)
        v = perturb(j.iloc[: , :-1], epsilon_value)
        print('v')
        print(v)
        w = estimate(v, epsilon_value)
        print(w)
        n = not_neg(w)
        print(n)
        run = rank(v, n, c)
        print(run)
        clfC = c45.C45()
        scores = cross_validate(clfC, v, y,
                                scoring=['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro',
                                         'recall_macro'])
        scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
        scoresDataFrame.drop(labels=['fit_time'], inplace=True)
        rowName = 'dtde' + '/' + x + '/'
        scoresDataFrame = scoresDataFrame.add_prefix(rowName)
        classifierDataFrame[epsilon_value] = scoresDataFrame
    classifierDataFrame.to_csv("./Experiments/test_run_" +
                               date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')




