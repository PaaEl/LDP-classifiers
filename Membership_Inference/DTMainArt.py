import math
import time as ti
from datetime import date

import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import DTTree
import DTTreeRAP
import pandas as pd
import DataPreprocessor
from Membership_Inference import shad
from pure_ldp.frequency_oracles import DEClient, DEServer, HEServer, HEClient, HadamardResponseServer, \
    HadamardResponseClient, LHServer, LHClient, UEServer, UEClient, RAPPORServer, RAPPORClient
epsilon = 1
d = 1
f = round(1/(0.5*math.exp(epsilon/2)+0.5), 2)
if f >= 1:
    f = 0.99
raps = RAPPORServer(f, 128, 8, d)
rapc = RAPPORClient(f, 128, raps.get_hash_funcs(), 8)
tree_a = DTTree
tree_rap = DTTreeRAP
shadow_nr =5
epsilon = 1
d = 1
des = DEServer(epsilon=epsilon, d=d)
dec = DEClient(epsilon=epsilon, d=d)
tree_a = DTTree
ldp_mechanism = {'de': (dec, des, tree_a)}
database_names=['adult','mushroom','iris','vote','car','nursery','spect','weightliftingexercises','htru']
epsilon_values=[0.1,1,5]
depth = [6 ]

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

'''Calculates precision and recall'''
def calc_precision_recall(predicted, actual, positive_value=1):
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
        client = a[0]
        tree = a[2]
        for xx in database_names:
            # getting the data in order
            b = DataPreprocessor.DataPreprocessor()
            X, y = b.get_data(xx)
            X, y = sklearn.utils.shuffle(X, y)
            X= X.reset_index(drop=True)
            X = X.astype('int')
            feat = list(X.columns)
            do = []
            for x in X.columns:
                do.append(max(X[x]) + 1)
            c = max(y) + 1
            # gets domainsize for each feature
            do = [gg * c for gg in do]

            classifierDataFrame = pd.DataFrame()
            for epsilon_value in epsilon_values:
                # encode and perturb the data
                X.insert(len(X.columns), 'label', y)
                j = encode(X, c)
                print(j)
                servers_l = tree.Tree()
                v = servers_l.perturb(j.iloc[:, :-1], epsilon_value, server, client, do)
                X = X.drop(columns=['label'])

                # to gather the statistics
                balanced_accuracy = []
                accuracy = []
                times = []
                f1 = []
                prec = []
                recall = []
                att_acc = []
                mema = []
                nonmema=[]
                # loop 10 times
                for i in range(1):
                    i += 1

                    # create the model to be attacked
                    clf = tree.Tree(attrNames=feat, depth=depth, ldpMechanismClient=client,
                                    ldpMechanismServer=server, epsilon_value=epsilon_value,
                                    domainSize=do, max=c)

                    # same but for rappor
                    # clf = tree.Tree(attrNames=feat, depth=depth, ldpMechanismClient=client,
                    #                                 ldpMechanismServer=server, epsilon_value=epsilon_value,
                    #                                 domainSize=do, max=c, tree=servers_l)

                    # divide the data to create shadow datasets and training and testing sets, shuffling happens later
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
                    x_target_test = x_target_test.reset_index(drop=True)
                    X_target_test = X_target_test.reset_index(drop=True)
                    x_target_train = x_target_train.reset_index(drop=True)
                    X_target_train = X_target_train.reset_index(drop=True)

                    # train the model to be attacked
                    clf.fit(X_target_train, y_target_train, x_target_train)
                    start = ti.time()

                    # predict the test set, to check model accuracy and get nontraining members to attack later
                    pre = clf.predict(X_target_test)
                    stop = ti.time()

                    # to gather the shadow datasets
                    nonmember_y = []
                    member_y = []
                    member_predictions=[]
                    nonmember_predictions =[]
                    member_X= X[0:0]
                    nonmember_X = X[0:0]

                    # create shadow models
                    for i in range(shadow_nr):
                        x_shadow, y_shadow, X_shadow= \
                            sklearn.utils.shuffle(x_shadow, y_shadow, X_shadow)
                        x_shadow = x_shadow.reset_index(drop=True)
                        X_shadow = X_shadow.reset_index(drop=True)
                        abc = shad.Shad_tree()
                        member_x, member_ya, member_predictionsa, nonmember_Xa, nonmember_ya, \
                        nonmember_predictionsa, member_Xa, nonmember_x = \
                        abc.create_shadow(X_shadow,x_shadow,y_shadow, shadow_nr, clf,X.iloc[:, :-1], y)

                        nonmember_y = nonmember_y + nonmember_ya
                        member_y = member_y + member_ya
                        member_predictions = member_predictions + member_predictionsa
                        nonmember_predictions = nonmember_predictions + nonmember_predictionsa
                        member_X= member_X.append(member_Xa)
                        nonmember_X = nonmember_X.append(nonmember_Xa)

                    member_X = member_X.reset_index(drop=True)
                    nonmember_X= nonmember_X.reset_index(drop=True)
                    nonmember_x = nonmember_x.reset_index(drop=True)

                    # to gather the attack models, 1 for each label in the model to be attacked
                    att_list = []

                    # here we create the attack models
                    for i in range(c):
                        memb = []
                        non_memb = []

                        # get all the records that have the label of the attack model
                        for j in range(len(member_predictions)):

                            if member_predictions[j] == i:
                                memb.append(j)
                        for j in range(len(nonmember_predictions)):

                            if nonmember_predictions[j] == i:
                                non_memb.append(j)

                        # these are all in the training set of the shadow models
                        dafX = member_X[member_X.index.isin(memb)]
                        memlist = [1] * len(dafX)

                        # these are all not in the training set of the shadow models
                        daf2X = nonmember_X[nonmember_X.index.isin(non_memb)]
                        nonmemlist = [0] * len(daf2X)

                        daf2X = daf2X.reset_index(drop=True)
                        dafX= dafX.reset_index(drop=True)
                        dafX = dafX.append(daf2X)
                        memlist= memlist + nonmemlist

                        # the attack model
                        model = DecisionTreeClassifier()
                        model.fit(dafX, memlist)
                        att_list.append(model)

                    # to gather the data
                    precis = []
                    recal = []
                    atac = []
                    memac = []
                    nonmemac = []

                    # using the attack models to predict in which category the training and
                    # testing data of the original model belongs
                    for i in range(c):
                        memb = []
                        non_memb = []
                        # get all the records that have the label of the attack model
                        for j in range(len(y_target_train)):

                            if y_target_train[j] == i:
                                memb.append(j)
                        for j in range(len(pre)):

                            if pre[j] == i:
                                non_memb.append(j)

                        # these are all in the training set of the original model
                        daf = X_target_train[X_target_train.index.isin(memb)]
                        daf = daf.reset_index(drop=True)

                        # these are not
                        daf2 = X_target_test[X_target_test.index.isin(non_memb)]
                        daf2 = daf2.reset_index(drop=True)

                        jaaaaaa = att_list[i].predict(daf)
                        jaaaaaa2 = att_list[i].predict(daf2)

                        # memacc holds the accuracy of the predictions of the training data, nonmemacc for the testing
                        memacc = np.count_nonzero(jaaaaaa == 1)/ len(daf)
                        nonmemacc = np.count_nonzero(jaaaaaa2 == 0)/len(daf2)

                        # total attack accuracy
                        acc = (memacc * len(X_target_train) + nonmemacc * len(X_target_test)) / (
                                    len(X_target_train) + len(X_target_test))

                        # getting precision and recall
                        pr, re = calc_precision_recall(np.concatenate((jaaaaaa,
                                                                jaaaaaa2)),
                                                    np.concatenate(
                                                        (np.ones(len(daf)), np.zeros(len(daf2)))),positive_value=i)

                        precis.append(pr)
                        recal.append(re)
                        atac.append(acc)
                        memac.append(memacc)
                        nonmemac.append(nonmemacc)
                    prec.append(sum(precis)/c)
                    att_acc.append(sum(atac) / c)
                    recall.append(sum(recal)/c)
                    mema.append(sum(memac)/c)
                    nonmema.append(sum(nonmemac) / c)









                    # gathering the results
                    balanced_accuracy.append(balanced_accuracy_score(y_target_test, pre))
                    accuracy.append(accuracy_score(y_target_test, pre))
                    times.append(stop - start)

                scores = {'score_time': times, 'test_accuracy': accuracy, 'test_balanced_accuracy': balanced_accuracy,
                          'precision': prec, 'recall': recall, 'attack accuracy': att_acc,
                          'memberattack accuracy': mema, 'nonmemberattack accuracy': nonmema}
                scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
                rowName = xxx + '/' + xx + '/' + 'dpth' + str(depth) + '/'
                scoresDataFrame = scoresDataFrame.add_prefix(rowName)
                classifierDataFrame[epsilon_value] = scoresDataFrame
            classifierDataFrame.to_csv("./Experiments/test_run_" +
                                       date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')