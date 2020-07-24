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
from analysis.trelis import Trelis
from dataset.dataset_generator import DatasetGenerator

training_size = 5000 
test_size = 100000
constellation_size = 2
snr = 10 
exponent_decaying_channel_const = 0.8 
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
trelis = Trelis(viterbi_with_gmm,dataset)
symbol_predictons = trelis.apply_to_dataset()
symbol_error_rate = np.mean(symbol_predictons != dataset.get_test_labels())
symbol_predictons[0:10]
dataset.get_test_labels()[0:10]
print(symbol_error_rate)


"""
network = ViterbiNet(dataset, constellation_size, channel_memory_length,
                     batch_size, epochs, learning_rate)
network.compile_model()
network.fit_model()
network.save_model(r"trained_models/viterbinet.h5")

print(network.predict(dataset.get_train_data()[0:3]))
"""
