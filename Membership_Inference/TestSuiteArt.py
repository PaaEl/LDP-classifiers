from datetime import date

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.naive_bayes import GaussianNB

from Daan.LDPLogReg import LDPLogReg
from Daan.LDPNaiveBayes import LDPNaiveBayes
from DataPreprocessor import DataPreprocessor
import pandas as pd
import time as ti


def calc_precision_recall(predicted, actual, positive_value):
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
    def __init__(self, database_names, epsilon_values=[0.1, 1, 2], classifiers=[LDPNaiveBayes(LDPid='LH')],
                 onehotencoded=False):
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
                allScoresDataFrame = pd.DataFrame()
                for classifier in self.classifiers:
                    classifierDataFrame = pd.DataFrame()
                    for epsilon_value in self.epsilon_values:
                        for i in range(1):
                            # to randomize
                            X, y = sklearn.utils.shuffle(X, y)

                            # to gather the statistics
                            balanced_accuracy = []
                            accuracy = []
                            times = []
                            prec = []
                            recall = []
                            att_acc = []
                            mema = []
                            nonmema = []
                            classifier.set_params(epsilon=epsilon_value)

                            # divide the data to create shadow datasets and training and testing sets
                            target_train_size = len(X) // 8
                            x_shadow = X[:target_train_size * 6]
                            y_shadow = y[:target_train_size * 6]
                            x_train = X[target_train_size * 6:target_train_size * 7]
                            y_train = y[target_train_size * 6:target_train_size * 7]
                            x_test = X[target_train_size * 7:]
                            y_test = y[target_train_size * 7:]
                            x_shadow = x_shadow.reset_index(drop=True)
                            x_train = x_train.reset_index(drop=True)
                            x_test = x_test.reset_index(drop=True)

                            # create the model to be attacked
                            clf = LDPNaiveBayes()

                            # same but for LR
                            # clf = LDPLogReg()

                            # train the model to be attacked
                            clf.fit(x_train, y_train)

                            # predict the test set, to check model accuracy and get nontraining members to attack later
                            start = ti.time()
                            pre = clf.predict(x_test)
                            stop = ti.time()

                            balanced_accuracy.append(balanced_accuracy_score(y_test, pre))
                            accuracy.append(accuracy_score(y_test, pre))
                            times.append(stop - start)

                            # to gather the shadow datasets
                            x_shadow_train_att = x_shadow[0:0]
                            x_shadow_test_att = x_shadow[0:0]
                            shad_pred = []
                            y_shadow_att = []

                            # create shadow models
                            clfAt = LDPNaiveBayes()
                            # clfAt = LDPLogReg()

                            for i in range(5):
                                x_shadow, y_shadow = sklearn.utils.shuffle(x_shadow, y_shadow)
                                x_shadow = x_shadow.reset_index(drop=True)
                                target_train_size = len(x_shadow) // 8
                                x_shadow_train = x_shadow[target_train_size * 4:]
                                y_shadow_train = y_shadow[target_train_size * 4:]
                                x_shadow_test = x_shadow[:target_train_size * 4]
                                clfAt.fit(x_shadow_train, y_shadow_train)
                                shad_pred = shad_pred + clfAt.predict(x_shadow_test)
                                # for LR
                                # shad_pred = shad_pred + clfAt.predict(x_shadow_test).values.tolist()
                                y_shadow_att = y_shadow_att + y_shadow_train.tolist()
                                x_shadow_train_att = pd.concat([x_shadow_train_att, x_shadow_train])
                                x_shadow_test_att = pd.concat([x_shadow_test_att, x_shadow_test])

                            # to gather the attack models, 1 for each label in the model to be attacked
                            att_list = []

                            # here we create the attack models
                            for i in range(max(y) + 1):
                                memb = []
                                non_memb = []

                                # get all the records that have the label of the attack model
                                for j in range(len(y_shadow_att)):

                                    if y_shadow_att[j] == i:
                                        memb.append(j)
                                for j in range(len(shad_pred)):

                                    if shad_pred[j] == i:
                                        non_memb.append(j)

                                # these are all in the training set of the shadow models
                                daf = x_shadow_train_att[x_shadow_train_att.index.isin(memb)]
                                memlist = [1] * len(daf)

                                # these are not
                                daf2 = x_shadow_test_att[x_shadow_test_att.index.isin(non_memb)]
                                # LR can't deal with only one class
                                if len(daf2)==0:
                                    nonmemlist = [0]
                                    daf2 = x_shadow_test_att.head(1)
                                else:
                                    nonmemlist = [0] * len(daf2)

                                dafx = daf.append(daf2)
                                memlist = memlist + nonmemlist

                                # the attack model
                                model = GaussianNB()
                                # model = LogisticRegression()
                                model.fit(dafx, memlist)
                                att_list.append(model)

                            # to gather the data
                            precis = []
                            recal = []
                            atac = []
                            memac = []
                            nonmemac = []
                            sumlen =[]

                            # using the attack models to predict in which category the training and
                            # testing data of the original model belongs
                            for i in range(max(y) + 1):
                                memb = []
                                non_memb = []

                                # get all the records that have the label of the attack model
                                for j in range(len(y_train)):

                                    if y_train[j] == i:
                                        memb.append(j)
                                for j in range(len(pre)):

                                    if pre[j] == i:
                                        non_memb.append(j)

                                # these are all in the training set of the original model
                                daf = x_train[x_train.index.isin(memb)]
                                daf = daf.reset_index(drop=True)

                                # these are not
                                daf2 = x_test[x_test.index.isin(non_memb)]
                                daf2 = daf2.reset_index(drop=True)
                                if len(daf2) == 0:
                                    daf2 = x_test.head(1)

                                jaaaaaa = att_list[i].predict(daf)
                                jaaaaaa2 = att_list[i].predict(daf2)

                                # memacc holds the accuracy of the predictions of the training data, nonmemacc for the testing
                                memacc = np.count_nonzero(jaaaaaa == 1) / len(daf)
                                nonmemacc = np.count_nonzero(jaaaaaa == 0) / len(daf2)

                                # total attack accuracy
                                acc = (memacc * len(x_train) + nonmemacc * len(x_test)) / (
                                        len(x_train) + len(x_test))

                                # getting precision and recall
                                pr, re = calc_precision_recall(np.concatenate((jaaaaaa,
                                                                               jaaaaaa2)),
                                                               np.concatenate(
                                                                   (np.ones(len(daf)), np.zeros(len(daf2)))),
                                                               positive_value=i)
                                vc = len(daf)+len(daf2)
                                sumlen.append(vc)
                                precis.append(pr*(vc))
                                recal.append(re*(vc))
                                atac.append(acc)
                                memac.append(memacc)
                                nonmemac.append(nonmemacc)
                            prec.append(sum(precis)  / (sum(sumlen)))
                            att_acc.append(sum(atac) / (max(y) + 1))
                            recall.append(sum(recal) /  (sum(sumlen)))
                            mema.append(sum(memac) / (max(y) + 1))
                            nonmema.append(sum(nonmemac) / (max(y) + 1))

                            # gathering the results
                            scores = {'score_time': times, 'test_accuracy': accuracy,
                                      'test_balanced_accuracy': balanced_accuracy,
                                      'precision': prec, 'recall': recall, 'attack accuracy': att_acc,
                                      'memberattack accuracy': mema, 'nonmemberattack accuracy': nonmema}
                            scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
                            rowName = str(classifier) + '/' + database_name + '/' + 'dpth' + '/'
                            scoresDataFrame = scoresDataFrame.add_prefix(rowName)
                            classifierDataFrame[epsilon_value] = scoresDataFrame
                    allScoresDataFrame = allScoresDataFrame.append(classifierDataFrame)
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
        scores.to_csv("./Experiments/test_run_" + date.today().__str__() + '.csv', mode='a', sep=';',
                      float_format='%.3f')