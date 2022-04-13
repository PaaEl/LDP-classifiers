import os
import pandas as pd
import json
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def get_data(self, database_name, onehotencoded=False):
        """ Process and return the X and y values
        Parameters
        ----------
        database_name : string,
                        Specifies the database name to be used. Should be the same as the .data filename in the Datasets folder
        onehotencoded : bool, default: False
                        Specifies if data needs to be one hot encoded

        Returns
        -------
        X, y : ndarray, shape (n_samples,)
                Returns X and y
        
        """
        self.X, self.y = self._get_raw_data(database_name)
        self._label_encode()
        if onehotencoded:
            self._one_hot_encode()
        self._bin_encode(10)
        self._clean_up()
        return self.X, self.y

    def _set_data_info(self, database_info):
        """ Sets the names of the different types of features

        Parameters
        ----------
        database_info : JSON format
            Specifies categorical_features, target_features and continuous_features
        
        Returns
        -------
        None
        """
        self.categorical_features = database_info['categorical_features']
        self.target_features = database_info['target_features']
        self.continuous_features = database_info['continuous_features']

    def _get_raw_data(self, database_name):
        """ Get the data from the data location

        Parameters
        ----------
        database_name : string
                        Specifies the database name. Should be the same as the .data filename in the Datasets folder
        
        Returns
        -------
        X, y : ndarray, shape (n_samples,)
            Returns the data split into X and y
        """
        database_info = self._get_database_info()[database_name]
        database_location = os.path.abspath(os.getcwd() + "/Datasets/" + database_name + ".data")
        self._set_data_info(database_info)

        raw_data = pd.read_csv(database_location)
        all_features = database_info["categorical_features"] + database_info["continuous_features"]
        X = raw_data.loc[:, all_features]
        y = raw_data[database_info["target_features"]]
        return X, y

    def _get_database_info(self):
        file = open("./Datasets/database_info.json", 'r')
        return json.load(file)

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
        self.y = y_encoder.fit_transform(self.y.values.ravel())

    def _one_hot_encode(self):
        """ One hot encode encode the categorical data

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        encoder = OneHotEncoder()
        encoder.fit(pd.DataFrame(self.X.loc[:, self.categorical_features]))
        columnNames = encoder.get_feature_names_out()
        tempXdata = pd.DataFrame(encoder.transform(pd.DataFrame(self.X.loc[:, self.categorical_features])).toarray(), columns=columnNames)
        self.X = self.X.drop(self.categorical_features, axis=1)
        self.X.loc[:,columnNames] = tempXdata

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
            encoder = KBinsDiscretizer(n_bins=n, encode="ordinal", strategy='uniform')
            self.X.loc[:,self.continuous_features] = encoder.fit_transform(self.X.loc[:, self.continuous_features])

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
