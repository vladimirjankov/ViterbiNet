"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
-----------------------------------
"""



from base_model import BaseModel
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D

class ViterbiNet(BaseModel):

    def __init__(self, dataset):
        super().__init__(dataset)


