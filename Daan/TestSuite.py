from LDPNaiveBayes import LDPNaiveBayes
from DataPreprocessor import DataPreprocessor
from sklearn.model_selection import cross_validate

class TestSuite():
    def __init__(self, epsilon_values=[0.1,1,2], classifiers=[LDPNaiveBayes(LDPid='LH')]):
        self.epsilon_values = epsilon_values
        self.classifiers = classifiers
        self.preprocessor = DataPreprocessor()

    """ Sets the specific parameters that will need to be evaluated
    Parameters
    ----------
    epsilon_values : {array} 
                     Float values representing the epsilon values
    classifiers :    {array}
                     Classifier objects

    Returns
    -------
    None
    """
    def set_params(self, epsilon_values, classifiers):
        self.epsilon_values = epsilon_values
        self.classifiers = classifiers

    """ Sets the specific parameters that will need to be evaluated
    Parameters
    ----------
    database_location : {string}
                        path to database location
    categorical_features : {array - string}
                           Names of the catergorical features
    target_feature : {array - string}
                           Names of the target feature
    continuous_features : {array - string}
                           Names of the continuous features

    Returns
    -------
    None
    """
    def set_database_params(self, database_location, categorical_features, target_feature, continuous_features=[]):
        self.preprocessor.set_data_info(database_location, categorical_features, target_feature, continuous_features)

    """ Runs the test suite for each value of the given parameters
    Parameters
    ----------
    None

    Returns
    -------
    None
    
    """
    def run(self):
        X, y = self.preprocessor.get_data()
        for classifier in self.classifiers:
            print(classifier)
            for epsilon_value in self.epsilon_values:
                classifier.set_params(epsilon=epsilon_value)
                scores = cross_validate(classifier, X, y, scoring=['balanced_accuracy','f1_macro', 'precision_macro', 'recall_macro'])
                self.print_scores(scores, epsilon_value)

    """ Prints the scores in a specific format
    Parameters
    ----------
    scores : {array}
             Score values accuracy, f1, precision and recall
    epsilon : {float}
              The epsilon value used to obtain the scores
              
    Returns
    -------
    None
    """
    def print_scores(self, scores, epsilon):
        print("Epsilon value: ", epsilon)
        print('-'*50)
        print("Accuracy: ", scores['test_balanced_accuracy'].mean())
        print("Recall: ", scores['test_recall_macro'].mean())
        print("Precision: ", scores['test_precision_macro'].mean())