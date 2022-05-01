from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array

class TemplateClassifier(BaseEstimator):
        """ TemplateClassifier
        Rename the file and class name to your own classifier name.

        """

        def __init__(self, epsilon=1, LDPid='DE'):
            """ Init function 
            Any parameter that the classifier takes should be mentioned here.
            For example: 
                epsilon value, or 
                ldp mechanism identifier

            Epsilon is a requirement and should NOT be removed.
            """
            self.epsilon = epsilon
            self.LDPid = LDPid

        def fit(self, X, y):
            """ fit function
            This function should take care of training the classifier on the supplied X and y data.
            
            It is required to return self
            """
            # Don't remove this step. Required to make sure X and y are in the correct format.
            X, y = check_X_y(X, y, accept_sparse=False, multi_output=True) 

            """
            Add your code here
            """

            # Don't remove following steps:
            self.is_fitted_ = True
            return self

        def predict(self, X):
            """ predict function
            This function should take care of predicting the values for the given X using the trained classifier.
            
            Should return an array of prediction values in the form of numerical coded values. 
            For example:
                [0,1,0,1,1,...etc]
            
            Returns
            -------
            y : ndarray, shape (n_samples,)
                Returns array of predictions
            """
            # Don't remove this step. Required to make sure X is in the correct format.
            X = check_array(X, accept_sparse=False)

            """
            Add your code here
            """

            return [] # Change to make own code return array.