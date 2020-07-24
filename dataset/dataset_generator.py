"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
-----------------------------------
"""

import numpy as np
import more_itertools as mit
from keras.utils import to_categorical

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

    def reshape_data(self, vector, memory):
        

        number_of_columns = vector.shape[0]
        matrix_representation = np.zeros((memory,number_of_columns))

        for row in range(0,memory):
            index = memory - row - 1
            matrix_representation[index][0:number_of_columns - index] = vector[index:]

        return matrix_representation

    def generate_data(self):
        
        self._fadding_channel = np.exp(self._exponent_decaying_channel_const * np.array([*range(0, self._channel_memory_length)]))
        self._combine_vector = np.array([pow(2, step) for step in range(0,self._channel_memory_length)])
        
        self._train_labels = np.random.randint(MINNIMAL_VALUE_DATA, self._constellation_size + 1, 
                                         self._training_size)
        self._test_labels = np.random.randint(MINNIMAL_VALUE_DATA, self._constellation_size + 1, 
                                        self._test_size)
        
        """
        self._train_labels = np.matmul(self._combine_vector, (np.transpose(np.array(list(mit.windowed(train_labels.ravel(),
                                       n = self._channel_memory_length)))) -1))

        
        
        self._test_labels =  np.matmul(self._combine_vector, (np.transpose(np.array(list(mit.windowed(test_labels.ravel(),
                                       n = self._channel_memory_length)))) -1 ))
        """
        data = 2 * (self._train_labels - 0.5 * (self._constellation_size + 1))
        data = self.reshape_data(data, self._channel_memory_length)
        
        #after deceying
        data = np.matmul(np.flip(self._fadding_channel,0), data)
        self._sigmaWdB = pow(10, -1 * self._snr / 10.0)
        self._train_data = data + np.sqrt(self._sigmaWdB) * np.random.randn(data.shape[0]) 
        
        data = 2 * (self._test_labels - 0.5 * (self._constellation_size + 1))
        data = self.reshape_data(data, self._channel_memory_length)
        #after deceying
        data = np.matmul(np.flip(self._fadding_channel,0), data)
        self._test_data = data + np.sqrt(self._sigmaWdB) * np.random.randn(data.shape[0]) 

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
        return (self._train_labels)

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

    def get_snr(self):
        """
        Returns snr
        :param: none
        :return snr
        """
        return self._snr

    def get_exponent_decaying_channel_const(self):
        """
        Returns exponent decaying channel const
        :param: none
        :return exponent decaying channel const
        """
        return self._exponent_decaying_channel_const

    def get_channel_memory_length(self):
        """
        Returns channel memory
        :param: none
        :return: channel memory length
        """
        return self._channel_memory_length    

    def get_constellation_size(self):
        """
        Return constellation size
        :param: none
        :return: constellation size
        """
        return self._constellation_size

    def get_fadding_channel(self):
        """
        Return fadding channel
        :param: none
        :return: fadding channel
        """
        return self._channel_memory_length     

