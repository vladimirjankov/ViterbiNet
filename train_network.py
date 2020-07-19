"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
-----------------------------------
"""

from model.viterbitnet import ViterbiNet
from dataset.dataset_generator import DatasetGenerator

training_size = 5000 
test_size = 100000
constellation_size = 2
snr = -6.0
exponent_decaying_channel_const = 0.1
channel_memory_length = 4 
batch_size = 27
epochs = 100
learning_rate = 0.01


dataset = DatasetGenerator(training_size, test_size, constellation_size, snr,
                           exponent_decaying_channel_const, channel_memory_length)

dataset.generate_data()

network = ViterbiNet(dataset, constellation_size, channel_memory_length,
                    batch_size, epochs, learning_rate)
network.compile_model()
network.fit_model()