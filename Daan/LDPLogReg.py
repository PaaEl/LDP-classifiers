import numpy as np
import pandas as pd
import random
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.special import expit
from scipy.sparse import issparse
from LDPMechanism import LDPMeanMechanism
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import accuracy_score 

class LDPLogReg(BaseEstimator):
    
    def __init__(self, epsilon=2, max_iter=10, learning_rate=2, LDPid='DU'):
        """ LDP Logisitc Regression estimator
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
        return "Logistic Regression Classifier"

    def fit(self, X, y):
        """A reference implementation of a fitting function.
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
        Returns
        -------
        None
        """
        X = check_array(X, accept_sparse=False)
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
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        score : float
                Accuracy score
        """
        prediction = self.predict(X)
        return accuracy_score(prediction, y)

    def _fit_single(self, X, y):
        tempWeights = [random.random() for _ in range(X.shape[1])]
        for _ in range(self.max_iter):
            # CALCULATE GRADIENT FOR EACH FOW OF DATA
            gradients = np.array(self._calculateGradient(X, y, tempWeights)).astype(float)
            # PERTURB THE GRADIENTS
            perturbedGradients = self._LDPMechanism.perturb(gradients, epsilon=self.epsilon/self.max_iter)
            # AGGREGATE AND GET MEAN FOR EACH FEATURE
            meanGradient = self._LDPMechanism.aggregate(perturbedGradients)
            # meanGradient = gradients.mean(axis=0)

            # USE MEAN TO UPDATE WEIGHTS
            tempWeights = tempWeights + self.learning_rate * meanGradient
            # REGULARIZATION STEP
            tempWeights = tempWeights - self.learning_rate * (0.6 / len(y)) * tempWeights     # TODO check optimal regression value. Now 0.6
        return tempWeights

    def _predictSingle(self, X, weights):
        # The function expit is a sigmoid function
        return expit(np.dot(X, weights))

    def _calculateGradient(self, X, y, weights):
        prediction = self._predictSingle(X, weights)
                
        gradient = pd.DataFrame(X).multiply((y - prediction),axis=0)

        # Clip the value of the gradient to fit [-1,1]
        clippedGradient = np.clip(gradient, -1, 1)
        return clippedGradient

    def _encode_y(self, y):
        encoder = OneHotEncoder()
        encoder.fit(pd.DataFrame(y))
        columnNames = encoder.get_feature_names_out()
        return pd.DataFrame(encoder.transform(pd.DataFrame(y)).toarray(), columns=columnNames)
