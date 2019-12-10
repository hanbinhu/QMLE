import logging
logger = logging.getLogger('qmle')
#import jax
#import jax.numpy as np
#from functools import partial
import numpy as np
import tensornetwork as tn
from itertools import product

class TTN(object):
    def __init__(self, image_size, channel, bond_data, bond_inner, bond_label, layer_channel):
        self.size = image_size
        self.channel = channel
        self.bond_data = bond_data
        self.bond_inner = bond_inner
        self.bond_label = bond_label
        self.layer_comb_channel = layer_channel

        self.layer_size = []
        lsize = self.size//2
        while lsize > 0:
            self.layer_size.append(lsize)
            lsize //= 2
        self.num_layers = len(self.layer_size)
        assert(self.layer_comb_channel <= self.num_layers)
        if self.num_layers == self.layer_comb_channel:
            self.layer_size.insert(self.num_layers, 1)
        else:
            self.layer_size.insert(self.layer_comb_channel,self.layer_size[self.layer_comb_channel]*2)
        self.num_layers += 1
        self.layer_type = [1]*self.layer_comb_channel # 1: With Ch, 0: Comb Ch, -1:W.O. Ch
        self.layer_type.append(0)
        self.layer_type += [-1]*(self.num_layers-self.layer_comb_channel-1)

        self.backend = 'numpy'
        #self.backend = 'jax'
        #self.jax_key = jax.random.PRNGKey(0)

        self.data_tensor_list = []
        for i in range(self.size):
            self.data_tensor_list.append([])
            for j in range(self.size):
                self.data_tensor_list[i].append([])
                for c in range(self.channel):
                    self.data_tensor_list[i][j].append(tn.Node(np.zeros((self.bond_data,)), backend=self.backend))

        self.label_tensor = tn.Node(np.zeros((self.bond_label,)), backend=self.backend)

        self.ttn_tensor_list = []
        for l in range(self.num_layers):
            self.ttn_tensor_list.append([])
            if l == 0:
                if self.layer_type[l] == 1:
                    size_info = (self.bond_data,)*4 + (self.bond_inner,)
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        for j in range(self.layer_size[l]):
                            self.ttn_tensor_list[l][i].append([])
                            for c in range(self.channel):
                                rnd_tensor = np.random.random(size_info)
                                #rnd_tensor = jax.random.uniform(self.jax_key, shape=size_info)
                                self.ttn_tensor_list[l][i][j].append(tn.Node(rnd_tensor, backend=self.backend))
                elif self.layer_type[l] == 0:
                    size_info = (self.bond_data,)*self.channel+(self.bond_inner,)
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        for j in range(self.layer_size[l]):
                            rnd_tensor = np.random.random(size_info)
                            #rnd_tensor = jax.random.uniform(self.jax_key, shape=size_info)
                            self.ttn_tensor_list[l][i].append(tn.Node(rnd_tensor, backend=self.backend))
                else:
                    raise RuntimeError
            elif l == self.num_layers-1:
                if self.layer_type[l] == -1:
                    size_info = (self.bond_inner,)*4+(self.bond_label,)
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        for j in range(self.layer_size[l]):
                            rnd_tensor = np.random.random(size_info)
                            #rnd_tensor = jax.random.uniform(self.jax_key, shape=size_info)
                            self.ttn_tensor_list[l][i].append(tn.Node(rnd_tensor, backend=self.backend))
                elif self.layer_type[l] == 0:
                    size_info = (self.bond_inner,)*self.channel+(self.bond_label,)
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        for j in range(self.layer_size[l]):
                            rnd_tensor = np.random.random(size_info)
                            #rnd_tensor = jax.random.uniform(self.jax_key, shape=size_info)
                            self.ttn_tensor_list[l][i].append(tn.Node(rnd_tensor, backend=self.backend))
                else:
                    raise RuntimeError
            else:
                if self.layer_type[l] == 1:
                    size_info = (self.bond_inner,)*4 + (self.bond_inner,)
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        for j in range(self.layer_size[l]):
                            self.ttn_tensor_list[l][i].append([])
                            for c in range(self.channel):
                                rnd_tensor = np.random.random(size_info)
                                #rnd_tensor = jax.random.uniform(self.jax_key, shape=size_info)
                                self.ttn_tensor_list[l][i][j].append(tn.Node(rnd_tensor, backend=self.backend))
                elif self.layer_type[l] == -1:
                    size_info = (self.bond_inner,)*4+(self.bond_inner,)
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        for j in range(self.layer_size[l]):
                            rnd_tensor = np.random.random(size_info)
                            #rnd_tensor = jax.random.uniform(self.jax_key, shape=size_info)
                            self.ttn_tensor_list[l][i].append(tn.Node(rnd_tensor, backend=self.backend))
                else:
                    size_info = (self.bond_inner,)*self.channel+(self.bond_inner,)
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        for j in range(self.layer_size[l]):
                            rnd_tensor = np.random.random(size_info)
                            #rnd_tensor = jax.random.uniform(self.jax_key, shape=size_info)
                            self.ttn_tensor_list[l][i].append(tn.Node(rnd_tensor, backend=self.backend))
        logger.debug("TTN initialization completed.")

    def connect_network(self):
        # disconnect everything
        for l in range(self.num_layers):
            if self.layer_type[l] == 1:
                for i, j, c in product(range(self.layer_size[l]), range(self.layer_size[l]), range(self.channel)):
                    for edge in self.ttn_tensor_list[l][i][j][c].get_all_nondangling():
                        edge.disconnect()
            else:
                for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                    for edge in self.ttn_tensor_list[l][i][j].get_all_nondangling():
                        edge.disconnect()

        # connect data
        if self.layer_type[0] == 1:
            for i, j, c in product(range(self.layer_size[0]), range(self.layer_size[0]), range(self.channel)):
                node = self.ttn_tensor_list[0][i][j][c]
                dnode_00 = self.data_tensor_list[2*i][2*j][c]
                dnode_01 = self.data_tensor_list[2*i][2*j+1][c]
                dnode_10 = self.data_tensor_list[2*i+1][2*j][c]
                dnode_11 = self.data_tensor_list[2*i+1][2*j+1][c]
                tn.connect(dnode_00[0], node[0])
                tn.connect(dnode_01[0], node[1])
                tn.connect(dnode_10[0], node[2])
                tn.connect(dnode_11[0], node[3])
        else:
            for i, j in product(range(self.layer_size[0]), range(self.layer_size[0])):
                node = self.ttn_tensor_list[0][i][j]
                for c in range(self.channel):
                    dnode = self.data_tensor_list[i][j][c]
                    tn.connect(dnode[0], node[c])

        # connect internal TTN
        for l in range(1, self.num_layers):
            if self.layer_type[l] == 1:
                for i, j, c in product(range(self.layer_size[l]), range(self.layer_size[l]), range(self.channel)):
                    node = self.ttn_tensor_list[l][i][j][c]
                    ttnode_00 = self.ttn_tensor_list[l-1][2*i][2*j][c]
                    ttnode_01 = self.ttn_tensor_list[l-1][2*i][2*j+1][c]
                    ttnode_10 = self.ttn_tensor_list[l-1][2*i+1][2*j][c]
                    ttnode_11 = self.ttn_tensor_list[l-1][2*i+1][2*j+1][c]
                    tn.connect(ttnode_00[-1], node[0])
                    tn.connect(ttnode_01[-1], node[1])
                    tn.connect(ttnode_10[-1], node[2])
                    tn.connect(ttnode_11[-1], node[3])
            elif self.layer_type[l] == -1:
                for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                    node = self.ttn_tensor_list[l][i][j]
                    ttnode_00 = self.ttn_tensor_list[l-1][2*i][2*j]
                    ttnode_01 = self.ttn_tensor_list[l-1][2*i][2*j+1]
                    ttnode_10 = self.ttn_tensor_list[l-1][2*i+1][2*j]
                    ttnode_11 = self.ttn_tensor_list[l-1][2*i+1][2*j+1]
                    tn.connect(ttnode_00[-1], node[0])
                    tn.connect(ttnode_01[-1], node[1])
                    tn.connect(ttnode_10[-1], node[2])
                    tn.connect(ttnode_11[-1], node[3])
            else:
                for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                    node = self.ttn_tensor_list[l][i][j]
                    for c in range(self.channel):
                        ttnode = self.ttn_tensor_list[l-1][i][j][c]
                        tn.connect(ttnode[-1], node[c])

    def connect_label(self):
        topnode = self.ttn_tensor_list[self.num_layers-1][0][0]
        tn.connect(topnode[-1], self.label_tensor[0])

    def update_input_tensor(self, x):
        for i, j, c in product(range(self.size), range(self.size), range(self.channel)):
            self.data_tensor_list[i][j][c].tensor = x[i,j,c,:]

    def update_label_tensor(self, y):
        self.label_tensor.tensor = y

    def get_update_size(self, node):
        return tuple(node.get_dimension(r) for r in range(node.get_rank()))

    def get_env_single_wo_channel_tensor(self, cl, ci, cj, x, y):
        self.update_input_tensor(x)
        self.update_label_tensor(y)
        self.connect_network()
        self.connect_label()

    def update_single_wo_channel_tensor(self, x, y, cl, ci, cj):
        num_sample = y.shape[0]
        update_size = self.get_update_size(self.ttn_tensor_list[cl][ci][cj])
        update_tensor = np.zeros(update_size)
        for n in range(num_sample):
            self.get_env_single_wo_channel_tensor(cl, ci, cj, x[n,:,:,:,:], y[n])
            
        #jax.vmap(partial(self.get_env_single_wo_channel_tensor, cl, ci, cj))(x, y)

    def update_single_w_channel_tensor(self, x, y, cl, ci, cj, cc):
        pass

    def update_single_comb_channel_tensor(self, x, y, cl, ci, cj):
        pass

    def train(self, x, y, epoch):
        for n in range(epoch):
            for l in range(self.num_layers):
                if self.layer_type[l] == -1:
                    for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                        self.update_single_wo_channel_tensor(x, y, l, i, j)
                        logger.debug(f"Done one wo tensor ({l},{i},{j}).")
                elif self.layer_type[l] == 0:
                    for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                        self.update_single_comb_channel_tensor(x, y, l, i, j)
                        logger.debug(f"Done one comb tensor ({l},{i},{j}).")
                else:
                    for i, j, c in product(range(self.layer_size[l]), range(self.layer_size[l]), range(self.channel)):
                        self.update_single_w_channel_tensor(x, y, l, i, j, c)
                        logger.debug(f"Done one w tensor ({l},{i},{j},{c}).")
            logger.debug(f'Epoch {n} completed.')

    def predict(self, x):
        n_sample = x.shape[-1]
        n_output = self.bond_label
        return np.random.rand(n_sample, n_output)
