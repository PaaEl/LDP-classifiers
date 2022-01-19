from Classifier import Classifier
import pandas as pd
from LDPMechanism import LDPMechanism, LDPid

class NBClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.classProbabilities = []
        self.featureProbabilities = []
    
    def fit(self, featureData, classData, epsilon, LDPid=LDPid.LH):
        super().fit(featureData,classData,epsilon, LDPid)
        self._encodeFeatures()
        self._perturb()
        self._calculateClassProbabilities()
        self._calculateFeatureProbabilities()

    # Connect the value of the classification to each feature value using:
    # featureValue * k + v
    # where k is the size of the classification domain and v is the actual classification value
    def _encodeFeatures(self):
        k = max(self.classData.unique())+1

        result = pd.DataFrame()

        for feature in self.features:
            dataColumn = self.featureData.loc[:,feature]
            result[feature] = dataColumn * k + self.classData

        return result

    # Perturb the data using the pure-LDP module.
    # The result is a list of server-objects for each feature colum that can be used to produce frequency estimates
    def _perturb(self):
        # Perturb features first
        servers = []
        encodedFeatures = self._encodeFeatures()
        for feature in self.features:
            d = max(encodedFeatures[feature].unique())+1

            LDPClient = self.LDPMechanism.client()
            LDPClient.update_params(self.epsilon, d)
            LDPServer = self.LDPMechanism.server()
            LDPServer.update_params(self.epsilon, d)

            # Privatise all the items in the current feature column
            tempColumn = encodedFeatures[feature].apply(lambda item : LDPClient.privatise(item+1))
            # Send all privatized item to the server object
            tempColumn.apply(lambda item : LDPServer.aggregate(item))
            servers.append(LDPServer)
        self.featureLDPServers = servers

        # Perturb classification values
        LDPClient = self.LDPMechanism.client()
        LDPClient.update_params(self.epsilon, d)
        LDPServer = self.LDPMechanism.server()
        LDPServer.update_params(self.epsilon, d)
        for row in self.classData:
            privateData = LDPClient.privatise(row+1)
            LDPServer.aggregate(privateData)
        self.classLDPServer = LDPServer

    # Calculate the probabilites per class
    def _calculateClassProbabilities(self):
        k = max(self.classData.unique()) + 1
        classificationSize = len(self.classData)
        estimates = [self.classLDPServer.estimate(i+1) for i in range(k)]
        estimates = [1 if estimate < 0 else estimate for estimate in estimates]
        self.classProbabilities = [estimates[i] / classificationSize for i in range(k)]

    # Calculate the probabilities for each encoded feature
    def _calculateFeatureProbabilities(self):
        k = max(self.classData.unique())+1
        # Go over every class value in it's domain
        for classValue in range(k):
            tempProbabilities = []
            # Go over every feature and calculate it't conditional probability given the classValue
            for index, feature in enumerate(self.features):
                d = max(self.featureData.loc[:, feature].unique())+1
                estimates = []
                for i in range(d):
                    featureIndex = i * k + classValue
                    try:
                        estimate = round(self.featureLDPServers[index].estimate(featureIndex+1))
                    except IndexError:
                        estimate = 1
                    freq = 1 if estimate < 0 else estimate
                    estimates.append(freq)
                sumEstimates = sum(estimates)
                tempProbabilities.append([estimates[i] / sumEstimates for i in range(d)])
            self.featureProbabilities.append(tempProbabilities)

    def predict(self, X_test, y_test):
        results = []
        for row in range(X_test.shape[0]):
            prediction = []
            for classVal in range(4): # 4 because the classification has domain size of 4
                classProb = self.classProbabilities[classVal]
                featureProb = 1
                for i, val in enumerate(X_test.iloc[row]):
                    featureProb = featureProb * self.featureProbabilities[classVal][i][val]
                prediction.append(classProb * featureProb)
            results.append(prediction.index(max(prediction)))
        return results