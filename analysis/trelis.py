"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 23.7.2020
-----------------------------------
"""

import numpy as np 

class Trelis(object):

    """
    Viterbi: Trelis and analisys 
    """
    def __init__(self, viterbi_with_gmm, dataset):
        """
        init terlis with Viterbi
        :param viterbi_with_gmm - viterbinet model 
        and PDF using GaussianMixture
        return none
        """
        self._viterbi_with_gmm = viterbi_with_gmm
        self._dataset = dataset
        self._trelis = None
        

    def apply_to_dataset(self, dataset = None):
        """
        Apply Viterbi detection based on calculations
        from viterbinet and pdf on evaluation data
        param: dataset = evaluation dataset. Default 
        value None
        return: recovered symbols probably 
        """
        if dataset == None: 
            dataset = self._dataset

        # Gets basic dataset properies 
        channel_memory_length = self._dataset.get_channel_memory_length()
        constellation_size = dataset.get_constellation_size()
        number_of_classes = pow(constellation_size, channel_memory_length)
        data_size = len(dataset.get_test_data())

        self._trelis = np.zeros((data_size, constellation_size))
        # generate trellis matrix
        for state in range(0, number_of_classes):
            index = np.mod(state -1, pow(dataset.get_constellation_size(),
                                        dataset.get_channel_memory_length() - 1))
            for tran in range(0, constellation_size):
                self._trelis[state][tran] = constellation_size * index + tran
        
        predicted_probabilities = self._viterbi_with_gmm.predict(dataset.get_test_data())
        log_priors = -1 * np.log(predicted_probabilities)
        states = np.zeros((number_of_classes,1))
        output = np.zeros((predicted_probabilities.shape[0],1))

        for data_point_index in range(0, data_size):
            next_state = np.zeros((number_of_classes, 1))
            for state in range(0,number_of_classes):
                temp = np.zeros((constellation_size,1))
                for tran in range(0,constellation_size):
                    temp[tran] = states[int(self._trelis[state][tran])] + log_priors[data_point_index][state]
                next_state[state] = np.min(temp)
            states = next_state
            index = np.where(states == np.min(states))[0]
            output[data_point_index] = np.mod(index[0],constellation_size) + 1

        return output.astype(int)

