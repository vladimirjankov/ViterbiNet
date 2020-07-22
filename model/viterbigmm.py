"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 21.7.2020
-----------------------------------
"""
import numpy as np

from model.viterbinet import ViterbiNet
from sklearn.mixture import GaussianMixture


class ViterbiGMM(object):


    def __init__(self, dataset, constellation_size, channel_memory_length,batch_size,
                 epochs, learning_rate, pre_train = False, load_path = None ):
        self._dataset = dataset
        self._constellation_size = constellation_size
        self._channel_memory_length = channel_memory_length
        self._number_of_classes = pow(self._constellation_size, self._channel_memory_length)
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._pre_train = pre_train
        self._network_model = ViterbiNet(dataset, constellation_size, 
                                         channel_memory_length, batch_size, epochs, 
                                         learning_rate)
        self._load_path = load_path

    def fit(self):
        """
        Fits the Viterbi neuralNetwork and fits 
        output PDF using GaussianMixture fitting
        :param none
        :return none
        """
        length = (self._dataset.get_train_data().shape[0])
        print(length)
        print(self._dataset.get_train_data().shape)
        # Computes output PDF using GMM fitting
        print("Training of Gaussian model in progress")
        self._gaussian_model  = GaussianMixture(n_components = self._number_of_classes)
        self._gaussian_model.fit(self._dataset.get_train_data().reshape(length,1))
        print("Training of Gaussian model done")
        # Check if there is model to be loaded 
        if(self._pre_train == False):
            self._network_model.compile_model()
            self._network_model.fit_model()
        else:
            self._network_model.load_model(self._load_path)


    def predict(self, data):
        """
        Predicts class of data_vector
        :param data_vector - vector of classes
        :return  recovered symbol
        """ 
        # Compute likelyhood
        likelyhood_function = self._network_model.predict(data)
        print(data.shape)
        data = data.reshape(len(data),1)
        # Compute output PDF
        out_pdf = self._gaussian_model.predict_proba(data)
        print(out_pdf)

        # ML-based log-likelihood computation
        y_likelyhood = np.multiply(likelyhood_function, out_pdf) * self._number_of_classes
        print(y_likelyhood)
        return y_likelyhood
        
            

""""
Ako imam i forward i backward passing 
forward : gledaj logicki 1111-0000 nije moguce
backward : matricu za forward transponuj :D
"""