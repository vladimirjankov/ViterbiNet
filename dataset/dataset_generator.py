"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
-----------------------------------
"""

import numpy as np

class DatasetGenerator(Object):
    """
    Class that generates dataset for ViterbiNet
    """

    def __init(self,training_size ,test_size):
        self._train_data = None
        self._train_labels = None
        self._test_data = None
        self._test_labels = None
        self._training_size = training_size
        self._test_size = test_size
        
    def generate_data():
        pass

    def get_train_data(self):
        """
        Returns training data
        :param none
        :return training data
        """
        return self._train_data

    def get_train_labels(self):
        """
        Returns training labels
        :param none
        :return training data
        """
        return self._train_labels 

    def get_test_data(self):
        """
        Returns test data
        :param none
        :return test data
        """
        return self._test_data

    def get_test_labels(self):
        """
        Returns test labels
        :param none
        :return test data
        """
        return self._test_data
