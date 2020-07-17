"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
-----------------------------------
"""


import numpy as np

from abc import ABC
from keras.models import Sequential


class BaseModel(object):
    def __init__(self,dataset):
        
        # Training and testing dataset
        self._dataset = dataset

        # Neural network
        self._model = Sequential()

		# Training time.
		self.train_time = 0

    


    

