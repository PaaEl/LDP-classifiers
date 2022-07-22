import math
from datetime import date

from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from tree import Tree
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import numpy as np
from pure_ldp.frequency_oracles import LHClient, LHServer, DEClient, DEServer
import DataPreprocessor

database_names=['adult','mushroom','iris','vote','car','nursery','spect','htru']
epsilon_values=[10]
depth = 4
# 'adult','mushroom','iris','vote','car','nursery','spect','weightliftingexercises','htru'
def hash_perturb(io):
    g = client_olh.privatise(io)
    return g
def encode(df, c):
    perturbed_df = pd.DataFrame()
    y = df.iloc[: , -1]
    # print('enc')
    # print(df)
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
    g = perturbed_df.iloc[:, :-1]
    g.insert(len(g.columns),'label',y)
    # print(g)
    return g

def decode(df, cat, c):
    df = pd.DataFrame(df)
    df.insert(len(df.columns),'label',cat)
    # print('decode')
    # print(df)
    perturbed_df = pd.DataFrame()
    for x in df.columns:
        tempColumn = df.loc[:, x] - df.iloc[: , -1]
        tempColumn = tempColumn.apply(lambda item: item / c)
        perturbed_df[x] = tempColumn
    # print(perturbed_df.iloc[:, :-1].astype('int'))
    return perturbed_df.iloc[:, :-1].astype('int')


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

def estimate(df, e, do):
    lis =[]
    i = 0
    # print(df.columns)
    # print(do)
    for x in df.columns:
        epsilon = e
        # d = max(df[x]) + 1
        # print('doi')
        # print(do[i])
        server_olh.update_params(epsilon, do[i])
        client_olh.update_params(epsilon, do[i])
        df.loc[:, x].apply(lambda g: server_olh.aggregate(g))
        li = []
        for j in range(0, do[i]):
            li.append(round(server_olh.estimate(j + 1)))
        lis.append(li)
        i +=1
    return lis

def not_neg(lis):
    t = [[j if j >0 else 0 for j in y] for y in lis]
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
    j=0
    while i < len(lis):
        # print('i')
        # print(i)
        # print(lis)
        # print(c)
        bb = sum(lis [i-1:i+c-1])
        # print(bb)
        fraction.append(bb/b)
        while j < (i+ c-1):
            # print(j)
            if bb == 0:
                bbb = 0
            else:
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

# iris = load_iris()
# # print(iris)
# clf = Tree(attrNames=iris.feature_names)
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)
# clf.fit(X_train, y_train)

for x in database_names:
    b = DataPreprocessor.DataPreprocessor()
    X, y = b.get_data(x)
    X = X.astype('int')

    # rslt_df = X['Percentage'] > 80
    #
    # print('\nResult dataframe :\n', rslt_df)
    feat = list(X.columns)
    print(feat)
    # print('uni')
    # print(X['odor'].value_counts())
    do = []
    for x in X.columns:
        do.append(max(X[x]) + 1)
    X.insert(len(X.columns),'label',y)
    c = max(y) + 1
    do = [gg * c for gg in do]
    # print('do')
    # print(do)
    epsilon = 10
    d = 10
    client_olh = DEClient(epsilon=epsilon, d=d)
    server_olh = DEServer(epsilon=epsilon, d=d)
    server_olh2 = DEServer(epsilon=epsilon, d=d)
    for epsilon_value in epsilon_values:
        j = encode(X, c)
        v = perturb(j.iloc[:, :-1], epsilon_value)
        # w = estimate(v, epsilon_value, do)
        # print('w')
        # print(w)
        clf = Tree(attrNames=feat, depth=depth, ldpMechanismClient=DEClient(epsilon=epsilon_value, d=d),
                   ldpMechanismServer=DEServer(epsilon=epsilon_value, d=d), epsilon_value=epsilon_value,
                   domainSize=do, max=c)
        # scores = cross_validate(clf, X.iloc[:, :-1], y,
        #                         scoring=['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro',
        #                                  'recall_macro'])
        # scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
        # scoresDataFrame.drop(labels=['fit_time'], inplace=True)
        # rowName = 'dtde' + '/' + x + '/'
        # scoresDataFrame = scoresDataFrame.add_prefix(rowName)
        # scoresDataFrame.to_csv("./Experiments/test_run_" +
        #                        date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')
        X_train, X_test, y_train, y_test = train_test_split(v, y, test_size=0.2)
        # print('split')
        # print(X_test)
        # print(y_test)
        # print('xiloc')
        # print(X.iloc[:, :-1])
        # print(X_train)
        # j = encode(X_train, c)
        # v = perturb(j, epsilon_value)
        # w = estimate(v, epsilon_value, do)
        # print(v)
        # print('w')
        # print(w)
        X_test = decode(X_test, y_test,c)
        clf.fit(X_train, y_train)
        # clf.predict(X_test)
        # print(f'Accuracy: {clf.score(X_test, y_test)}')
        fff= balanced_accuracy_score(y_test, clf.predict(X_test))
        ff = accuracy_score(y_test, clf.predict(X_test))
        print(fff)
        print(ff)
