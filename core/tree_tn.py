import logging
logger = logging.getLogger('qmle')
import numpy as np
import tensornetwork as tn

class TTN(object):
    def __init__(self, image_size, channel, bond_data, bond_inner, bond_label):
        self.num_image_size = image_size
        self.num_image_channel = channel
        self.dim_bond_data = bond_data
        self.dim_bond_inner = bond_inner
        self.dim_bond_label = bond_label

    def train(self, x, y, epoch):
        for i in range(epoch):
            logger.debug(f'Epoch {i} completed.')

    def predict(self, x):
        n_sample = x.shape[-1]
        n_output = self.dim_bond_label
        return np.random.rand(n_sample, n_output)
