"""
-----------------------------------
@author: Vladimir Jankov
@email: vladimir.jankov@outlook.com
@date: 17.7.2020
-----------------------------------
"""

import numpy as np

from model.viterbinet import ViterbiNet
from model.viterbigmm import ViterbiGMM
from dataset.dataset_generator import DatasetGenerator


training_size = 5000 
test_size = 100000
constellation_size = 2
snr = -6
exponent_decaying_channel_const = 0.8 #0.5
channel_memory_length = 4 
batch_size = 25
epochs = 100
learning_rate = 0.01


dataset = DatasetGenerator(training_size, test_size, constellation_size, snr,
                           exponent_decaying_channel_const, channel_memory_length)

dataset.generate_data()

viterbi_with_gmm = ViterbiGMM(dataset, constellation_size, channel_memory_length,
                              batch_size, epochs, learning_rate)
viterbi_with_gmm.fit()
viterbi_with_gmm.predict(dataset.get_test_data())

"""
network = ViterbiNet(dataset, constellation_size, channel_memory_length,
                     batch_size, epochs, learning_rate)
network.compile_model()
network.fit_model()
network.save_model(r"trained_models/viterbinet.h5")

print(network.predict(dataset.get_train_data()[0:3]))
"""
