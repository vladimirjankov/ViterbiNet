"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
-----------------------------------
"""
import time

from tensorflow import keras
from model.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, InputLayer, ReLU, Softmax
from model.base_model import BaseModel
NUMBER_OF_HIDDEN_UNITS = 100

class ViterbiNet(BaseModel):

    def __init__(self, dataset, constellation_size, channel_memory_length,
                 batch_size, epochs, learning_rate):
        super().__init__(dataset)
        self._constellation_size = constellation_size
        self._channel_memory_length = channel_memory_length
        self._number_of_classes = self._constellation_size * self._channel_memory_length
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate

    def evaluate_model(self):
        """
        Evaluates the ViterbiNet model.
        :param none
        :return none
        """
        self._scores = self._dataset.evaluate(x = self._dataset.get_test_data(),
                                             y = self._dataset.get_test_labels())
        
        print("Test loss : ", self._scores[0])
        print("Test accuracy : ", self._scores[1])
        

    def define_sequential_model(self):
        """
        Sets the ViterbiNet arhitecture.
        1 x 100 , 100 x 50, 50 x num

        """
        print(self._dataset.get_train_data().shape)
        print(self._dataset.get_train_labels())
        
        self._model.add(LSTM(NUMBER_OF_HIDDEN_UNITS, input_shape=(5000,4), return_sequences=False))
        self._model.add(Dense(100, activation='sigmoid'))
        self._model.add(Dense(50, activation='sigmoid'))
        self._model.add(Dense(self._number_of_classes, activation='softmax'))

    
    def compile_model(self):
        """
		Configures the ViterbiNet model.
		:param none
		:return none
		"""
        self.define_sequential_model()
        self._model.compile(optimizer="adam", loss='mse')

    def fit_model(self):
        """
        Trains the ViterbitNet model.
        :param none
        :return none
        """

        start_time = time.time()
        
        
        print("Training of ViterbitNet in progress")
        self.history = self._model.fit( x = self._dataset.get_train_data(), 
                                        y = self._dataset.get_train_labels(),
                                        batch_size = self._batch_size,
                                        epochs  = self._epochs)

        end_time = time.time()
        self._train_time = end_time - start_time
        print("Training of ViterbitNet done")








