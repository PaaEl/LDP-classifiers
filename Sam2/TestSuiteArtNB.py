from datetime import date
from Daan.LDPNaiveBayes import LDPNaiveBayes
from DataPreprocessor import DataPreprocessor
from sklearn.model_selection import cross_validate
import pandas as pd

class TestSuite():
    def __init__(self, database_names, epsilon_values=[0.1,1,2], classifiers=[LDPNaiveBayes(LDPid='DE')], onehotencoded=False):
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
        for database_name in self.database_names:
            X, y = self.preprocessor.get_data(database_name, onehotencoded=self.onehotencoded)
            allScoresDataFrame = pd.DataFrame()
            for classifier in self.classifiers:
                classifierDataFrame = pd.DataFrame()
                for epsilon_value in self.epsilon_values:
                    classifier.set_params(epsilon=epsilon_value)
                    scores = cross_validate(classifier, X, y, scoring=['accuracy', 'balanced_accuracy','f1_macro',
                                                                       'precision_macro', 'recall_macro'])

                    scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
                    scoresDataFrame.drop(labels=['fit_time', 'score_time'], inplace=True)
                    rowName = classifier.__str__() + '/' + database_name + '/'
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
        scores.to_csv("./Experiments/test_run_" + date.today().__str__() + '.csv', mode='a', sep=';', float_format='%.3f')