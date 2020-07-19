"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
-----------------------------------
"""

import numpy as np
import more_itertools as mit

MINNIMAL_VALUE_DATA = 1

class DatasetGenerator(object):
    """
    Class that generates dataset for ViterbiNet
    """

    def __init__(self,training_size ,test_size, constellation_size, snr, 
               exponent_decaying_channel_const, channel_memory_length):
        self._train_data = None
        self._train_labels = None
        self._test_data = None
        self._test_labels = None
        self._fadding_channel = None
        self._training_size = training_size
        self._test_size = test_size
        self._constellation_size = constellation_size
        self._snr = snr
        self._exponent_decaying_channel_const = exponent_decaying_channel_const
        self._channel_memory_length = channel_memory_length
       
        
    def generate_data(self):
        
        self._fadding_channel = np.exp(self._exponent_decaying_channel_const * np.array([*range(0, self._channel_memory_length)]))
        self._combine_vector = np.array([pow(2, step) for step in range(0,self._channel_memory_length)])
        train_labels = np.random.randint(MINNIMAL_VALUE_DATA, self._constellation_size + 1, 
                                         self._training_size + self._channel_memory_length -1)
        test_labels = np.random.randint(MINNIMAL_VALUE_DATA, self._constellation_size + 1, 
                                        self._test_size + self._channel_memory_length -1)
        self._train_labels = np.matmul(self._combine_vector, (np.transpose(np.array(list(mit.windowed(train_labels.ravel(),
                                       n = self._channel_memory_length)))) -1))

        
        
        self._test_labels =  np.matmul(self._combine_vector, (np.transpose(np.array(list(mit.windowed(test_labels.ravel(),
                                       n = self._channel_memory_length)))) -1 ))
        
        data = 2 * (train_labels - 0.5 * (self._constellation_size + 1))

        data = np.transpose(np.array(list(mit.windowed(data.ravel(),
                            n = self._channel_memory_length))))
        #after deceying
        data = np.matmul(np.flip(self._fadding_channel,0), data)
        self._sigmaWdB = pow(10, -1 * self._snr / 10.0)
        self._train_data = data + np.sqrt(self._sigmaWdB) * np.random.randn(data.shape[0]) 
        

    def get_train_data(self):
        """
        Returns training data
        :param none
        :return training data
        """
        return self._train_data.reshape(5000,1,1)

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
        return self._test_labels
