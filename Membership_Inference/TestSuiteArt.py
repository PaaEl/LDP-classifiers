from datetime import date

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.naive_bayes import GaussianNB

from Daan.LDPLogReg import LDPLogReg
from Sam2.LDPNaiveBayesArt import LDPNaiveBayes
from Sam2.LDPNaiveBayesArtShad import LDPNaiveBayes as LP
from DataPreprocessor import DataPreprocessor
from sklearn.model_selection import cross_validate
import pandas as pd
import time as ti


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

class TestSuite():
    def __init__(self, database_names, epsilon_values=[0.1,1,2], classifiers=[LDPNaiveBayes(LDPid='LH')], onehotencoded=False):
        """
        Parameters
        ----------
        database_names : array of strings
                         Specifies the databases to be tested on
        epsilon_values : array of floats
                         The epsilon values to be tested on
        classifiers : array of Classifier objects
                      The classifiers to be used. Should be deriving from SKlearn BaseEstimator.
        """
        self.database_names = database_names
        self.epsilon_values = epsilon_values
        self.classifiers = classifiers
        self.onehotencoded = onehotencoded
        self.preprocessor = DataPreprocessor()

    
    def set_params(self, epsilon_values, classifiers, onehotencoded=False):
        """ Sets the specific parameters that will need to be evaluated
        Parameters
        ----------
        epsilon_values : {array} 
            Float values representing the epsilon values
        classifiers :    {array}
            Classifier objects
        onehotencoded : {bool}, default=False
            Bool that indicates if One Hot Encoding is necessary

        Returns
        -------
        None
        """
        self.epsilon_values = epsilon_values
        self.classifiers = classifiers
        self.onehotencoded = onehotencoded

    
    def run(self):
        """ Runs the test suite for each value of the given parameters
        Parameters
        ----------
        None

        Returns
        -------
        None
        
        """
        for i in range(1):
            for database_name in self.database_names:
                X, y = self.preprocessor.get_data(database_name, onehotencoded=self.onehotencoded)
                print(X)
                allScoresDataFrame = pd.DataFrame()
                for classifier in self.classifiers:
                    classifierDataFrame = pd.DataFrame()
                    for epsilon_value in self.epsilon_values:
                        for i in range(10):
                            X, y = sklearn.utils.shuffle(X, y)
                            print(i)
                            balanced_accuracy = []
                            accuracy = []
                            times = []
                            f1 = []
                            prec = []
                            recall = []
                            att_acc = []
                            mema = []
                            nonmema = []
                            classifier.set_params(epsilon=epsilon_value)
                            target_train_size = len(X) // 8
                            x_shadow = X[:target_train_size * 6]
                            y_shadow = y[:target_train_size * 6]

                            x_train = X[target_train_size * 6:target_train_size * 7]
                            y_train = y[target_train_size * 6:target_train_size * 7]
                            x_test = X[target_train_size * 7:]
                            y_test = y[target_train_size * 7:]
                            print('ddddvdsdvdpl,;')
                            print(x_shadow)
                            x_shadow = x_shadow.reset_index(drop=True)
                            print(x_shadow)
                            print(y_shadow)
                            # y_shadow = y_shadow.reset_index(drop=True)
                            # x_shadow_test = x_shadow_test.reset_index(drop=True)
                            y_shadow_test = y[target_train_size * 3:target_train_size * 6]
                            x_train = x_train.reset_index(drop=True)
                            # y_train = y_train.reset_index(drop=True)
                            x_test = x_test.reset_index(drop=True)
                            # y_test = y_test.reset_index(drop=True)
                            print('ddvsdvs')
                            print(target_train_size)
                            print(X)
                            print(x_train)
                            clf = LDPNaiveBayes()
                            # clf = LDPLogReg()
                            clf.fit(x_train, y_train)
                            start = ti.time()
                            pre = clf.predict(x_test)
                            stop = ti.time()
                            balanced_accuracy.append(balanced_accuracy_score(y_test, pre))
                            accuracy.append(accuracy_score(y_test, pre))
                            times.append(stop - start)
                            clfAt = LDPNaiveBayes()
                            # clfAt = LDPLogReg()
                            x_shadow_train_att= x_shadow[0:0]
                            x_shadow_test_att = x_shadow[0:0]
                            shad_pred = []
                            y_shadow_att = []

                            for i in range(5):
                                x_shadow, y_shadow = sklearn.utils.shuffle(x_shadow, y_shadow)
                                x_shadow = x_shadow.reset_index(drop=True)
                                target_train_size = len(x_shadow) // 8
                                x_shadow_train = x_shadow[target_train_size * 4:]
                                y_shadow_train = y_shadow[target_train_size * 4:]
                                print('bvmb')
                                print(y_shadow_train)
                                x_shadow_test = x_shadow[:target_train_size * 4]
                                y_shadow_test = y_shadow[:target_train_size * 4]
                                clfAt.fit(x_shadow_train, y_shadow_train)
                                shad_pred = shad_pred + clfAt.predict(x_shadow_test)
                                y_shadow_att = y_shadow_att + y_shadow_train.tolist()
                                x_shadow_train_att = pd.concat([x_shadow_train_att, x_shadow_train])
                                x_shadow_test_att = pd.concat([x_shadow_test_att, x_shadow_test])

                            att_list = []
                            for i in range(max(y) + 1):
                                memb = []
                                non_memb = []

                                for j in range(len(y_shadow_att)):

                                    if y_shadow_att[j] == i:
                                        memb.append(j)
                                for j in range(len(shad_pred)):

                                    if shad_pred[j] == i:
                                        non_memb.append(j)
                                daf = x_shadow_train_att[x_shadow_train_att.index.isin(memb)]
                                print('kijk')
                                print(x_shadow_train_att)
                                print(daf)
                                memlist = [1] * len(daf)
                                daf2 = x_shadow_test_att[x_shadow_test_att.index.isin(non_memb)]
                                nonmemlist = [0] * len(daf2)
                                dafx = daf.append(daf2)
                                memlist = memlist + nonmemlist
                                model = GaussianNB()
                                # model = LogisticRegression()
                                model.fit(dafx, memlist)
                                att_list.append(model)
                            print('dsssss')
                            print(att_list)
                            precis = []
                            recal = []
                            atac = []
                            memac = []
                            nonmemac = []
                            for i in range(max(y) + 1):
                                memb = []
                                non_memb = []

                                for j in range(len(y_train)):

                                    if y_train[j] == i:
                                        memb.append(j)
                                for j in range(len(y_test)):

                                    if y_test[j] == i:
                                        non_memb.append(j)
                                daf = x_train[x_train.index.isin(memb)]
                                # print('kijk')
                                print(x_train)
                                print(memb)
                                # print(daf)
                                # memlist = [1] * len(daf)
                                daf2 = x_test[x_test.index.isin(non_memb)]
                                dafx = daf.append(daf2)
                                print('saaa')
                                print(daf)
                                print(daf2)
                                jaaaaaa = att_list[i].predict(daf)
                                jaaaaaa2 = att_list[i].predict(daf2)
                                print('avds')
                                print(len(daf) + len(daf2))
                                print(jaaaaaa)
                                print(jaaaaaa2)
                                print(np.count_nonzero(jaaaaaa == 1))
                                # memacc = jaaaaaa.count(1) / len(daf)
                                # nonmemacc = jaaaaaa2.count(0) / len(daf2)
                                memacc = np.count_nonzero(jaaaaaa == 1) / len(daf)
                                nonmemacc = np.count_nonzero(jaaaaaa == 0) / len(daf2)
                                print(memacc)
                                print(nonmemacc)
                                print('bb')
                                acc = (memacc * len(x_train) + nonmemacc * len(x_test)) / (
                                        len(x_train) + len(x_test))
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
                                                                   (np.ones(len(daf)), np.zeros(len(daf2)))),
                                                               positive_value=i)
                                print(pr, re)
                                precis.append(pr)
                                recal.append(re)
                                atac.append(acc)
                                memac.append(memacc)
                                nonmemac.append(nonmemacc)
                            prec.append(sum(precis) / (max(y) + 1))
                            att_acc.append(sum(atac) / (max(y) + 1))
                            recall.append(sum(recal) / (max(y) + 1))
                            mema.append(sum(memac) / (max(y) + 1))
                            nonmema.append(sum(nonmemac) / (max(y) + 1))
                            scores = {'score_time': times, 'test_accuracy': accuracy,
                                      'test_balanced_accuracy': balanced_accuracy,
                                      'precision': prec, 'recall': recall, 'attack accuracy': att_acc,
                                      'memberattack accuracy': mema, 'nonmemberattack accuracy': nonmema}
                            # scores = cross_validate(classifier, X, y, scoring=['accuracy', 'balanced_accuracy','f1_macro', 'precision_macro', 'recall_macro'])
                            # scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
                            # scoresDataFrame.drop(labels=['fit_time', 'score_time'], inplace=True)
                            # rowName = classifier.__str__() + '/' + database_name + '/'
                            # scoresDataFrame = scoresDataFrame.add_prefix(rowName)
                            # classifierDataFrame[epsilon_value] = scoresDataFrame
                            scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
                            rowName = str(classifier) + '/' + database_name + '/' + 'dpth' + '/'
                            scoresDataFrame = scoresDataFrame.add_prefix(rowName)
                            classifierDataFrame[epsilon_value] = scoresDataFrame
                    allScoresDataFrame = allScoresDataFrame.append(classifierDataFrame)
                print(allScoresDataFrame)
                self.print_scores(allScoresDataFrame)

    
    def print_scores(self, scores):
        """ Prints the scores in a specific format to a csv file
        Parameters
        ----------
        scores : array, float
            Score values accuracy, f1, precision and recall

        Returns
        -------
        None
        """
        scores.to_csv("./Experiments/test_run_" + date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')