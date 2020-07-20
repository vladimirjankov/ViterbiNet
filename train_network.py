"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
-----------------------------------
"""

import numpy as np

from model.viterbitnet import ViterbiNet
from dataset.dataset_generator import DatasetGenerator


training_size = 5000 
test_size = 100000
constellation_size = 2
snr = -6
exponent_decaying_channel_const = 0.1
channel_memory_length = 4 
batch_size = 25
epochs = 100
learning_rate = 0.01


dataset = DatasetGenerator(training_size, test_size, constellation_size, snr,
                           exponent_decaying_channel_const, channel_memory_length)

dataset.generate_data()
print(dataset.get_train_data().shape)
print(dataset.get_train_labels().shape)

network = ViterbiNet(dataset, constellation_size, channel_memory_length,
                     batch_size, epochs, learning_rate)
network.compile_model()
network.fit_model()
network.save_model(r"trained_models/viterbinet.h5")

print(network.predict(dataset.get_train_data()[0:3]))
