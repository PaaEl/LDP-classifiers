import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.special import expit
from scipy.sparse import issparse
from Daan.LDPMechanism import LDPMeanMechanism
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import accuracy_score 

class LDPLogReg(BaseEstimator):
    
    def __init__(self, epsilon=2, max_iter=25, learning_rate=1, LDPid='DU'):
        """ LDP Logistic Regression estimator
        Parameters
        ----------
        max_iter    : {int}
                    The maximum amount of iterations done
        epsilon     : {int}
                    The epsilon (privacy) value. Lower is more private
        learning_rate : {int}
                    The learning rate of the logistic regession classifier. Controls the step size of the gradient.
        Returns
        -------
        None
        """
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.LDPid = LDPid
    
    def __str__(self):
        return "Logistic Regression Classifier_" + self.LDPid

    def fit(self, X, y):
        """The reference implementation of the fitting function.
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
                "Not supporting sparse data"
            )
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True)
        X = np.c_[X, np.ones(X.shape[0])]

        self._LDPMechanism = LDPMeanMechanism(self.LDPid)
        # Hot encode y
        y = self._encode_y(y)
        self._weights = []
        for column in y:
            self._weights.append(self._fit_single(X, y[column]))

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """ Predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns array of predictions
        """
        X = check_array(X, accept_sparse=False)
        X = np.c_[X, np.ones(X.shape[0])]
        check_is_fitted(self, 'is_fitted_')
        localPredictions = pd.DataFrame()
        for index, weightColumn in enumerate(self._weights):
            localPredictions[index] = self._predictSingle(X, weightColumn)
        return localPredictions.apply(lambda x : list(x).index(max(x)), axis=1)

    def score(self, X, y):
        """ Default scoring function
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            The target values 
        Returns
        -------
        score : float
                Accuracy score
        """
        prediction = self.predict(X)
        return accuracy_score(prediction, y)

    def _fit_single(self, X, y):
        """ Fit a classifier on a single row of classification values
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples)
            The target values 
        Returns
        -------
        tempWeights : {numpy-array}
                      Weights of the predict function
        """
        tempWeights = np.zeros(X.shape[1])
        learning_rate = self.learning_rate
        
        for _ in range(self.max_iter):
            # Make a prediction with the current set of weights
            predictions = self._predictSingle(X, tempWeights)
            # Calculate the gradients for each individual row
            gradients = np.array(self._calculateGradient(X, y, predictions)).astype(float)
            
            # Perturb all the gradients. Divide epsilon by number of iterations
            perturbedGradients = self._LDPMechanism.perturb(gradients, epsilon=self.epsilon/self.max_iter)
            # Aggregate and extract the mean from the perturbed gradients
            meanGradient = self._LDPMechanism.aggregate(perturbedGradients)

            # USE MEAN TO UPDATE WEIGHTS
            tempWeights = tempWeights + learning_rate * meanGradient
            # REGULARIZATION STEP
            tempWeights = tempWeights - learning_rate * (0.6 / len(y)) * tempWeights     # TODO check optimal regularization value. Now 0.6
            # DECREASE LEARNING RATE
            learning_rate = learning_rate - (self.learning_rate / self.max_iter)
        return tempWeights

    def _predictSingle(self, X, weights):
        """ Prediction based on a set of weights and feature data
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.
        weights : array-like, shape (n_samples)
                  The set of weights to calculate the prediction
        Returns
        -------
        predictions : {numpy-array}
            The set of predictions
        """
        # The function expit is a sigmoid function
        return expit(np.dot(X, weights))

    def _calculateGradient(self, X, y, predictions):
        """ Calculate the gradient based on the predications.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples)
            The target values 
        predictions : array-like, shape (n_samples)
                      The predictions made on te training input samples
        Returns
        -------
        gradients : array-like, shape (n_samples)
                    The gradient of each predication
        """
        gradient = pd.DataFrame(X).multiply((y - predictions),axis=0)

        # Clip the value of the gradient to fit [-1,1]
        clippedGradient = np.clip(gradient, -1, 1)
        return clippedGradient

    def _calculateCost(self, X, y, predictions):
        """ Calculate the cost of the predictions made
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples)
            The target values 
        predictions : array-like, shape (n_samples)
                      The predictions made on te training input samples
        Returns
        -------
        cost : array-like, shape (n_samples)
               The cost of each predication
        """
        return ((-y) * np.log(predictions) + (1-y) * np.log(1 - predictions))

    def _encode_y(self, y):
        """ One hot encode the target samples
        Parameters
        ----------
        y : array-like, shape (n_samples)
            The target values 
        Returns
        -------
        encoded_y : array-like, shape (y x n_samples)
                    The y split into separate columns one hot encoded.
        
        """
        encoder = OneHotEncoder()
        encoder.fit(pd.DataFrame(y))
        columnNames = encoder.get_feature_names_out()
        return pd.DataFrame(encoder.transform(pd.DataFrame(y)).toarray(), columns=columnNames)