"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
-----------------------------------
"""
from base_model import BaseModel
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, InputLayer, ReLU, Softmax

NUMBER_OF_HIDDEN_UNITS = 100

class ViterbiNet(BaseModel):

    def __init__(self, dataset, input_size, constellation_size, channel_memory_length):
        super().__init__(dataset)
        self._model.add()
        self._input_size = input_size
        self._constellation_size = constellation_size
        self._channelgithu_memory_length = channel_memory_length
        self._number_of_classes = self._constellation_size * self._channel_memory_length
    def fit_model(self):
        """
        Trains the NeuralNetwork
        :param none
        :return none
        """
        pass

    def evaluate_model(self):
        """
        Evaluates the NeuralNetwork model.
        :param none
        :return none
        """
        pass

    def define_sequential_model(self):
        """
        Sets the ViterbiNet arhitecture.
        1 x 100 , 100 x 50, 50 x num

        """
        self._model.add(InputLayer(shape=(4,)))
        self._model.add(LSTM(NUMBER_OF_HIDDEN_UNITS,return_sequences=False,return_state=True))
        self._model.add(Dense(100, activation='sigmoid'))
        self._model.add(Dense(50, activation='sigmoid'))
        self._model.add(Dense(self._number_of_classes, activation='softmax'))
        self._model.compile(optimizer='adam', loss='mse')







