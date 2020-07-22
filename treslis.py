"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
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

        channel_memory_length = self._dataset.get_channel_memory_length()

        number_of_classes = pow(dataset.get_constellation_size(),
                                dataset.get_channel_memory_length())

        constellation_size = dataset.get_constellation_size()
        data_size = len(dataset.get_test_data())
        self._trelis = np.zeros((data_size, constellation_size))
        # generate trellis matrix
        for state in range(0, number_of_classes):
            index = np.mod(state -1, pow(dataset.get_constellation_size(),
                                        dataset.get_channel_memory_length() - 1))
            for tran in range(0, constellation_size):
                self._trelis(state,tran) = constellation_size * index + tran

        for data_point_index in range(0, data_size):
            
        #### to do 
        # apply Viterbi 

""""
Ako imam i forward i backward passing 
forward : gledaj logicki 1111-0000 nije moguce
backward : matricu za forward transponuj :D
"""

        
        

