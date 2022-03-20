import pandas as pd
import numpy as np
from LDPMechanism import LDPFreqMechanism
from sklearn.base import BaseEstimator
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score 
from sklearn.utils.validation import check_X_y, check_array

class LDPNaiveBayes(BaseEstimator):
    def __init__(self, epsilon=1, LDPid='DE'):
        self.epsilon = epsilon
        self.LDPid = LDPid
    
    def __str__(self):
        return "Naive Bayes Classifier"

    def fit(self, X, y):
        if issparse(X):
            raise ValueError(
                "Not supporting sparse data."
            )
        if isinstance(y[0], complex):
            raise ValueError(
                "Complex data not supported"
            )
        X, y = check_X_y(X, y, multi_output=True)
        self._classProbabilities = []
        self._featureProbabilities = []
        self._LDPMechanism = LDPFreqMechanism(self.LDPid)
        self._uniqueClassValues = max(np.unique(y))
        self._n_features = len(X.T)
        self._perturb(X, y)
        self._calculateClassProbabilities(y)
        self._calculateFeatureProbabilities(X)
        return self
    """
    Connect the value of the classification to each feature value using:
    featureValue * k + v
    where k is the size of the classification domain and v is the actual classification value
    """
    def _encodeFeatures(self, X, y):
        self._k = self._uniqueClassValues+1

        result = pd.DataFrame()

        for n_feature in range(self._n_features):
            dataColumn = X.T[n_feature]
            # encodedColumn = dataColumn * self._k + y
            # result = pd.concat([result, pd.DataFrame(encodedColumn, columns=[n_feature])],axis=1)
            result[n_feature] = dataColumn * self._k + y
        
        return result.astype(int)
    """ 
    Perturb the data using the pure-LDP module.
    The result is a list of server-objects for each feature colum that can be used to produce frequency estimates
    """
    def _perturb(self, X, y):
        # Perturb features first
        servers = []
        encodedFeatures = self._encodeFeatures(X, y)
        for n_feature in range(self._n_features):
            d = int(max(encodedFeatures[n_feature].unique()))+1

            LDPClient = self._LDPMechanism.client()
            LDPClient.update_params(self.epsilon, d)
            LDPServer = self._LDPMechanism.server()
            LDPServer.update_params(self.epsilon, d)

            # Privatise all the items in the current feature column
            tempColumn = encodedFeatures[n_feature].apply(lambda item : LDPClient.privatise(item+1))
            # Send all privatized item to the server object
            tempColumn.apply(lambda item : LDPServer.aggregate(item))
            servers.append(LDPServer)
        self._featureLDPServers = servers

        # Perturb classification values
        LDPClient = self._LDPMechanism.client()
        LDPClient.update_params(self.epsilon, d)
        LDPServer = self._LDPMechanism.server()
        LDPServer.update_params(self.epsilon, d)
        for row in y:
            privateData = LDPClient.privatise(row+1)
            LDPServer.aggregate(privateData)
        self._classLDPServer = LDPServer

    """
    Calculate the probabilites per class
    """
    def _calculateClassProbabilities(self, y):
        classificationSize = len(y)
        estimates = [self._classLDPServer.estimate(i+1) for i in range(self._k)]
        estimates = [1 if estimate <= 0 else estimate for estimate in estimates]
        self._classProbabilities = [estimates[i] / classificationSize for i in range(self._k)]

    """
    Calculate the probabilities for each encoded feature
    """
    def _calculateFeatureProbabilities(self, X):
        # Go over every class value in it's domain
        for classValue in range(self._k):
            tempProbabilities = []
            # Go over every feature and calculate it't conditional probability given the classValue
            for index in range(self._n_features):
                d = int(max(np.unique(X.T[index]))+1)
                estimates = np.ones_like(np.arange(d))
                # Go over each unique value in this feature column. The number of unique values is d.
                for i in range(d):
                    featureIndex = i * self._k + classValue
                    try:
                        estimate = round(self._featureLDPServers[index].estimate(featureIndex+1))
                    except IndexError:
                        estimate = 1
                    freq = 1 if estimate <= 0 else estimate
                    estimates[i] = freq
                sumEstimates = sum(estimates)
                tempProbabilities.append([estimates[i] / sumEstimates for i in range(d)])
            self._featureProbabilities.append(tempProbabilities)

    """
    Predict the classification based on the test set.
    """
    def predict(self, X):
        X = check_array(X)
        results = []
        for row in range(X.shape[0]):
            prediction = []
            for classVal in range(self._uniqueClassValues+1):
                classProb = self._classProbabilities[classVal]
                featureProb = 1
                for i, val in enumerate(X[row]):
                    featureProb = featureProb * self._featureProbabilities[classVal][i][int(val)]
                prediction.append(classProb * featureProb)
            results.append(prediction.index(max(prediction)))
        return results

    def score(self, X, y):
        prediction = self.predict(X)
        return accuracy_score(prediction, y)