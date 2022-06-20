import pandas as pd
import numpy as np
from Daan.LDPMechanism import LDPFreqMechanism
from sklearn.base import BaseEstimator
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score 
from sklearn.utils.validation import check_X_y, check_array

class LDPNaiveBayes(BaseEstimator):
    def __init__(self, epsilon=1, LDPid='DE'):
        self.epsilon = epsilon
        self.LDPid = LDPid
    
    def __str__(self):
        return "Naive Bayes Classifier_" + self.LDPid

    def fit(self, X, y):
        """The implementation of the fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
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
        self._n_features = X.shape[1]
        self._perturb(X, y)
        self._calculateClassProbabilities(y)
        self._calculateFeatureProbabilities(X)
        return self

    def _connectFeaturesWithClass(self, X, y):
        """ Connect the value of the classification to each feature value using:
        featureValue * k + v
        where k is the size of the classification domain and v is the actual classification value

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        connected_features : array-like
                             connection of all features
        """
        self._k = self._uniqueClassValues+1

        connectedTemp = X.T * self._k + y
        result = pd.DataFrame(connectedTemp.T)
        
        return result.astype(int)
    
    def _perturb(self, X, y):
        """ 
        Perturb the data using the pure-LDP module.
        The result is a list of server-objects for each feature colum that can be used to produce frequency estimates

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        None
            
        """
        # Perturb features first
        servers = []
        encodedFeatures = self._connectFeaturesWithClass(X, y)
        for n_feature in range(self._n_features):
            d = int(max(encodedFeatures[n_feature].unique()))+1

            params = {"epsilon":self.epsilon, "d":d}
            LDPServer = self._LDPMechanism.server(params)

            # TEMP so change entire if statement!!!!
            hashMechanisms = ['HR', 'RAPPOR']
            if self._LDPMechanism.LDPid in hashMechanisms:             
                LDPClient = self._LDPMechanism.client(params, LDPServer)
            else:
                LDPClient = self._LDPMechanism.client(params)
            # TEMP so change entire if statement!!!!

            # Privatise all the items in the current feature column
            tempColumn = encodedFeatures[n_feature].apply(lambda item : LDPClient.privatise(item+1))
            # Send all privatized item to the server object
            tempColumn.apply(lambda item : LDPServer.aggregate(item))
            servers.append(LDPServer)
        self._featureLDPServers = servers

        # Perturb classification values
        params = {"epsilon":self.epsilon, "d":d}
        LDPServer = self._LDPMechanism.server(params)

        # TEMP so change entire if statement!!!!
        if self._LDPMechanism.LDPid in hashMechanisms:             
            LDPClient = self._LDPMechanism.client(params, LDPServer)
        else:
            LDPClient = self._LDPMechanism.client(params)
        # TEMP so change entire if statement!!!!
        
        for row in y:
            privateData = LDPClient.privatise(row+1)
            LDPServer.aggregate(privateData)
        self._classLDPServer = LDPServer

    def _calculateClassProbabilities(self, y):
        """
        Calculate the probabilites per class
        Parameters
        ----------
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        None
        """
        classificationSize = len(y)
        estimates = [self._classLDPServer.estimate(i+1, suppress_warnings=True) for i in range(self._k)]
        estimates = [1 if estimate <= 0 else estimate for estimate in estimates]
        self._classProbabilities = [estimates[i] / classificationSize for i in range(self._k)]

    def _calculateFeatureProbabilities(self, X):
        """
        Calculate the probabilities for each encoded feature
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        None
        """
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
                        estimate = round(self._featureLDPServers[index].estimate(featureIndex+1, suppress_warnings=True))
                    except IndexError:
                        estimate = 1
                    freq = 1 if estimate <= 0 else estimate
                    estimates[i] = freq
                sumEstimates = sum(estimates)
                tempProbabilities.append([estimates[i] / sumEstimates for i in range(d)])
            self._featureProbabilities.append(tempProbabilities)

    def predict(self, X):
        """
        Predict the classification based on the test set.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        predictions : array-like
                      predictions
        """
        X = check_array(X)
        results = []
        for row in range(X.shape[0]):
            prediction = []
            for classVal in range(self._uniqueClassValues+1):
                classProb = self._classProbabilities[classVal]
                featureProb = 1
                for i, val in enumerate(X[row]):
                    try:
                        featureProb = featureProb * self._featureProbabilities[classVal][i][int(val)]
                    except IndexError:
                        featureProb = featureProb
                prediction.append(classProb * featureProb)
            results.append(prediction.index(max(prediction)))
        return results