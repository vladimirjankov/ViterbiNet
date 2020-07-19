"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
-----------------------------------
"""

#from model.viterbitnet import ViterbiNet
from dataset.dataset_generator import DatasetGenerator

dataset = DatasetGenerator(training_size = 5000 ,test_size = 1000, constellation_size = 2, snr = 1, 
               exponent_decaying_channel_const = 1 , channel_memory_length = 4)

dataset.generate_data()

print(dataset.get_train_data())

#netWork = ViterbiNet()
