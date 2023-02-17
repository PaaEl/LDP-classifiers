import math
import time as ti
from datetime import date

import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier
import shadow_models
import DTTreeArt
import DTTreeHR
import DTTreeRAP
import pandas as pd
from sklearn.model_selection import train_test_split
import DataPreprocessor
from Sam2 import shad
from pure_ldp.frequency_oracles import DEClient, DEServer, HEServer, HEClient, HadamardResponseServer, \
    HadamardResponseClient, LHServer, LHClient, UEServer, UEClient, RAPPORServer, RAPPORClient

shadow_nr =3
epsilon = 1
d = 1
des = DEServer(epsilon=epsilon, d=d)
dec = DEClient(epsilon=epsilon, d=d)
tree_a = DTTreeArt
ldp_mechanism = {'de': (dec, des, tree_a)}
database_names=['weightliftingexercises','htru']
epsilon_values=[0.1,1,5]
depth = [4 ]

# 'de': (dec, des, tree_a), 'olh': (lhc, lhs, tree_a), 'hr': (hrc, hrs, tree_hr),
#                  'he': (hec, hes, tree_a), 'oue': (uec, ues, tree_a), 'rap': (rapc, raps, tree_rap)
# 0.01,0.1,0.5,1,2,3,
# 'adult','mushroom','iris','vote','car','nursery','spect','weightliftingexercises','htru'


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


def calc_precision_recall(predicted, actual, positive_value=1):
    print('pred')
    print(len(predicted))
    print(len(actual))
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall



for xxxx in depth:
    depth = xxxx
    for xxx in ldp_mechanism:
        a = ldp_mechanism[xxx]
        server = a[1]
        print(server)
        client = a[0]
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
            print('do')
            print(do)
            classifierDataFrame = pd.DataFrame()
            for epsilon_value in epsilon_values:
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                j = encode(X, c)
                print(j)
                servers_l = tree.Tree()
                v = servers_l.perturb(j.iloc[:, :-1], epsilon_value, server, client, do)
                print(v)
                print(X)
                # print(X)
                # print(v)
                balanced_accuracy = []
                accuracy = []
                times = []
                f1 = []
                prec = []
                recall = []
                att_acc = []
                # ten times and get the average
                for i in range(10):
                    i += 1
                    print(i)
                    clf = tree.Tree(attrNames=feat, depth=depth, ldpMechanismClient=client,
                                    ldpMechanismServer=server, epsilon_value=epsilon_value,
                                    domainSize=do, max=c)
                    target_train_size = len(v) // 4
                    x_shadow = v[:target_train_size * 3]
                    y_shadow = y[:target_train_size * 3]
                    x_target_test = v[target_train_size * 3:]
                    y_target_test = y[target_train_size * 3:]
                    X_shadow = X[:target_train_size * 3]
                    X_target_test = X[target_train_size * 3:]
                    x_target_train = x_target_test[len(x_target_test) // 2:]
                    y_target_train = y_target_test[len(y_target_test)// 2:]
                    X_target_train = X_target_test[len(X_target_test) //2:]
                    x_target_test = x_target_test[:len(x_target_test)//2]
                    y_target_test = y_target_test[:len(y_target_test)// 2]
                    X_target_test = X_target_test[:len(X_target_test)// 2]
                    clf2 = tree.Tree(attrNames=feat, depth=depth, ldpMechanismClient=client,
                                     ldpMechanismServer=server, epsilon_value=epsilon_value,
                                     domainSize=do, max=c)
                    # clf2.fit(X_shadow,y_shadow,x_shadow)
                    # print(x_target_test)
                    # clf.predict(x_target_test)
                    #                     print('adasa')
                    # print(x_target_test)
                    # print(y_target_test.size)
                    # print(X_target_test)
                    # y_target_test = y_target_test.reset_index(drop=True)
                    x_target_test = x_target_test.reset_index(drop=True)
                    X_target_test = X_target_test.reset_index(drop=True)
                    x_target_train = x_target_train.reset_index(drop=True)
                    X_target_train = X_target_train.reset_index(drop=True)
                    # print(X_target_test)
                    # X_train1, X_test1, y_train1, y_test1 = train_test_split(X.iloc[:, :-1], y, test_size=0.8)
                    # print(X_train1)
                    print('dsgdsacacc')
                    print(len(X_target_train))
                    print(len(y_target_train))
                    print(len(x_target_train))
                    print(X_target_train)
                    print(y_target_train)
                    print(x_target_train)
                    clf.fit(X_target_train, y_target_train, x_target_train)
                    start = ti.time()
                    pre = clf.predict(x_target_test)
                    stop = ti.time()




                    abc = shad.Shad_tree()
                    member_x, member_y, member_predictions, nonmember_x, nonmember_y, nonmember_predictions = \
                        abc.create_shadow(X_shadow,x_shadow,y_shadow, shadow_nr, clf2,X.iloc[:, :-1], y)
                    print(member_x, member_y, member_predictions, nonmember_x, nonmember_y, nonmember_predictions)
                    print(nonmember_y)
                    print(nonmember_predictions)
                    member_x_arr = member_x.to_numpy()
                    nonmember_x_arr = nonmember_x.to_numpy()
                    member_predictions_arr = np.array(member_predictions)
                    nonmember_predictions_arr = np.array(nonmember_predictions)
                    member_y_arr = np.array(member_y)
                    nonmember_y_arr = np.array(nonmember_y)
                    print('dsgds')
                    print(member_x)
                    print(type(member_x))
                    print(type(member_predictions_arr))
                    print(member_x_arr)
                    print(member_y)
                    print(type(nonmember_x))
                    print(type(nonmember_predictions_arr))
                    print(type(member_y))
                    print(type(nonmember_y))
                    print(len(member_x))
                    print(len(member_predictions_arr))
                    print(len(nonmember_x))
                    print(len(nonmember_predictions_arr))
                    print(len(member_y))
                    print(len(nonmember_y))

                    att_list = []
                    for i in range(max(y) +1):
                        memb = []
                        non_memb = []

                        for j in range(len(member_y)):

                            if member_y[j] == i:
                                memb.append(j)
                        for j in range(len(nonmember_y)):

                            if nonmember_y[j] == i:
                                non_memb.append(j)
                        daf = member_x[member_x.index.isin(memb)]
                        print('kijk')
                        print(member_x)
                        print(daf)
                        memlist = [1] * len(daf)
                        daf2 = nonmember_x[nonmember_x.index.isin(non_memb)]
                        nonmemlist = [0] * len(daf2)
                        dafx = daf.append(daf2)
                        memlist= memlist + nonmemlist
                        model = RandomForestClassifier()
                        model.fit(dafx, memlist)
                        att_list.append(model)
                    print('dsssss')
                    print(att_list)
                    precis = []
                    recal = []
                    atac = []
                    for i in range(max(y) +1):
                        memb = []
                        non_memb = []

                        for j in range(len(y_target_train)):

                            if y_target_train[j] == i:
                                memb.append(j)
                        for j in range(len(y_target_test)):

                            if y_target_test[j] == i:
                                non_memb.append(j)
                        daf = x_target_train[x_target_train.index.isin(memb)]
                        # print('kijk')
                        # print(member_x)
                        # print(daf)
                        # memlist = [1] * len(daf)
                        daf2 = x_target_test[x_target_test.index.isin(non_memb)]
                        dafx = daf.append(daf2)
                        # print('saaa')
                        # print(daf)
                        # print(daf2)
                        jaaaaaa = att_list[i].predict(daf)
                        jaaaaaa2 = att_list[i].predict(daf2)
                        print('avds')
                        print(len(daf)+len(daf2))
                        memacc = np.count_nonzero(jaaaaaa == 1)/ len(daf)
                        nonmemacc = np.count_nonzero(jaaaaaa2 == 1)/len(daf2)
                        print(memacc)
                        print(nonmemacc)
                        print('bb')
                        acc = (memacc * len(x_target_train) + nonmemacc * len(x_target_test)) / (
                                    len(x_target_train) + len(x_target_test))
                        print(acc)
                        print(len(jaaaaaa))
                        print(len(jaaaaaa2))
                        # memlist = [1] * len(daf)
                        # nonmemlist = [0] * len(daf2)
                        # memlist= memlist + nonmemlist
                        # print(len(memlist))
                        # print('saaaaaa')
                        # member_acc = sum(jaaaaaa) / len(daf)
                        # print(member_acc)
                        # member_acc = sum(jaaaaaa2) / len(daf2)
                        # # print(member_acc)
                        # print(calc_precision_recall(np.concatenate((jaaaaaa,
                        #                                         jaaaaaa2)),
                        #                             np.concatenate(
                        #                                 (np.ones(len(daf)), np.zeros(len(daf2)))),positive_value=i))
                        pr, re = calc_precision_recall(np.concatenate((jaaaaaa,
                                                                jaaaaaa2)),
                                                    np.concatenate(
                                                        (np.ones(len(daf)), np.zeros(len(daf2)))),positive_value=i)
                        print(pr,re)
                        precis.append(pr)
                        recal.append(re)
                        atac.append(acc)
                    prec.append(sum(precis)/(max(y)+1))
                    att_acc.append(sum(atac) / (max(y) + 1))
                    recall.append(sum(recal)/(max(y)+1))
                    print(prec)
                    print(recall)









                    # gathering the results
                    balanced_accuracy.append(balanced_accuracy_score(y_target_test, pre))
                    accuracy.append(accuracy_score(y_target_test, pre))
                    # f1.append(f1_score(y_train1, pre, average='weighted'))
                    # prec.append(precision_score(y_train1, pre, average='weighted'))
                    # recall.append(recall_score(y_train1, pre, average='weighted'))
                    times.append(stop - start)

                scores = {'score_time': times, 'test_accuracy': accuracy, 'test_balanced_accuracy': balanced_accuracy,
                          'precision': prec, 'recall': recall, 'attack accuracy': att_acc}
                scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
                rowName = xxx + '/' + xx + '/' + 'dpth' + str(depth) + '/'
                scoresDataFrame = scoresDataFrame.add_prefix(rowName)
                classifierDataFrame[epsilon_value] = scoresDataFrame
            classifierDataFrame.to_csv("./Experiments/test_run_" +
                                       date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')