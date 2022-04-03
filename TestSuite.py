from datetime import date
from Daan.LDPNaiveBayes import LDPNaiveBayes
from DataPreprocessor import DataPreprocessor
from sklearn.model_selection import cross_validate
import pandas as pd

class TestSuite():
    def __init__(self, database_name, epsilon_values=[0.1,1,2], classifiers=[LDPNaiveBayes(LDPid='LH')]):
        self.database_name = database_name
        self.epsilon_values = epsilon_values
        self.classifiers = classifiers
        self.preprocessor = DataPreprocessor(database_name)

    
    def set_params(self, epsilon_values, classifiers, onehotencoded=False):
        """ set_params: Sets the specific parameters that will need to be evaluated
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
        X, y = self.preprocessor.get_data(onehotencoded=self.onehotencoded)
        allScoresDataFrame = pd.DataFrame()
        for classifier in self.classifiers:
            classifierDataFrame = pd.DataFrame()
            for epsilon_value in self.epsilon_values:
                classifier.set_params(epsilon=epsilon_value)
                scores = cross_validate(classifier, X, y, scoring=['balanced_accuracy','f1_macro', 'precision_macro', 'recall_macro'])
                scoresDataFrame = pd.DataFrame.from_dict(scores).mean()
                scoresDataFrame.drop(labels=['fit_time', 'score_time'], inplace=True)
                rowName = classifier.__str__() + '/' + self.database_name + '/'
                scoresDataFrame = scoresDataFrame.add_prefix(rowName)
                classifierDataFrame[epsilon_value] = scoresDataFrame
            allScoresDataFrame = allScoresDataFrame.append(classifierDataFrame)
        self.print_scores(allScoresDataFrame)

    
    def print_scores(self, scores):
        """ print_scores: Prints the scores in a specific format to a csv file
        Parameters
        ----------
        scores : {array}
            Score values accuracy, f1, precision and recall

        Returns
        -------
        None
        """
        scores.to_csv("./Experiments/test_run_" + date.today().__str__() + '.csv', sep=';', float_format='%.3f')