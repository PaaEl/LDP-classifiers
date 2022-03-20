import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_location='', categorical_features=[], target_features=[], continuous_features=[]):
        self.data_location = data_location
        self.categorical_features = categorical_features
        self.target_features = target_features
        self.continuous_features = continuous_features
    
    """ Process and return the X and y values
    Parameters
    ----------
    None

    Returns
    -------
    X, y : ndarray, shape (n_samples,)
            Returns X and y
    
    """
    def get_data(self):
        self.X, self.y = self._get_raw_data()
        self._label_encode()
        self._bin_encode(10)
        self._clean_up()
        return self.X, self.y

    def set_data_info(self, data_location, categorical_features, target_features, continuous_features=[]):
        self.__init__(data_location, categorical_features, target_features, continuous_features)

    def _get_raw_data(self):
        """ Get the data from the data location

        Parameters
        ----------
        None
        
        Returns
        -------
        X, y : ndarray, shape (n_samples,)
            Returns the data split into X and y
        """
        raw_data = pd.read_csv(self.data_location)
        all_features = self.categorical_features + self.continuous_features
        X = raw_data.loc[:, all_features]
        y = raw_data[self.target_features]
        return X, y

    def _label_encode(self):
        """ Label encode the categorical data into numerical values

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        X_encoder = OrdinalEncoder()
        y_encoder = LabelEncoder()
        self.X.loc[:,self.categorical_features] =  X_encoder.fit_transform(self.X.loc[:, self.categorical_features])

        encoder = OneHotEncoder()
        encoder.fit(pd.DataFrame(self.X.loc[:, self.categorical_features]))
        columnNames = encoder.get_feature_names_out()
        tempX = pd.DataFrame(encoder.transform(pd.DataFrame(self.X.loc[:, self.categorical_features])).toarray(), columns=columnNames)
        self.X = self.X.drop(self.categorical_features, axis=1)
        self.X.loc[:,columnNames] = tempX
        self.y = y_encoder.fit_transform(self.y.values.ravel())

    def _bin_encode(self, n):
        """ Bin continuous data into n numer of bins

        Parameters
        ----------
        n : The number of bins
        
        Returns
        -------
        None
        """
        if (self.continuous_features):
            encoder = KBinsDiscretizer(n_bins=n, encode="ordinal", strategy='quantile')
            self.X.loc[:,self.continuous_features] =  encoder.fit_transform(self.X.loc[:, self.continuous_features])

    """ Clean the data by resetting index and randomizing the order

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
    def _clean_up(self):
        self.X, _, self.y, _ = train_test_split(self.X, self.y, random_state=32)
        self.X = self.X.reset_index(drop=True)
