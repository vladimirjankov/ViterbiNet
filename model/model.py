"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
-----------------------------------
"""



from base_model import BaseModel
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, input, ReLU, Softmax

class ViterbiNet(BaseModel):

    def __init__(self, dataset, input_size, constellation_size, channel_memory_length):
        super().__init__(dataset)
        self._model.add()
        self._input_size = input_size
        self._constellation_size = constellation_size
        self._channel_memory_length = channel_memory_length
        self._number_of_classes = self._constellation_size * self._channel_memory_length

    def fit_model(self):
        """
        Trains the neuralNetwork
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
        Defines the ViterbiNet arhitecture.


        """
        self._model.add(Dense())
        self._model.add(LSTM() )
        self._model.add(ReLU)
        self._model.compile(optimizer='adam', loss='mse')







