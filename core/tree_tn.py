import logging
logger = logging.getLogger('qmle')
#import jax
#import jax.numpy as np
#from functools import partial
import numpy as np
import copy
import tncontract as tn
from .contract import *
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

        self.flag_contract = []
        self.contracted = [0]
        self.ttn_tensor_list = []
        for l in range(self.num_layers):
            self.ttn_tensor_list.append([])
            self.flag_contract.append([])
            self.contracted.append([])
            if l == 0:
                if self.layer_type[l] == 1:
                    size_info = (self.bond_data,)*4 + (self.bond_inner,)
                    tensor_label = ["D00","D01","D10","D11","U"]
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        self.flag_contract[l].append([])
                        self.contracted[l+1].append([])
                        for j in range(self.layer_size[l]):
                            self.ttn_tensor_list[l][i].append([])
                            self.flag_contract[l][i].append([])
                            self.contracted[l+1][i].append([])
                            for c in range(self.channel):
                                rnd_tensor = np.random.random(size_info)
                                self.ttn_tensor_list[l][i][j].append(tn.Tensor(rnd_tensor, labels=tensor_label))
                                self.flag_contract[l][i][j].append(False)
                                self.contracted[l+1][i][j].append(0)
                elif self.layer_type[l] == 0:
                    size_info = (self.bond_data,)*self.channel+(self.bond_inner,)
                    tensor_label = ["C"+str(i) for i in range(self.channel)]+["U"]
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        self.flag_contract[l].append([])
                        self.contracted[l+1].append([])
                        for j in range(self.layer_size[l]):
                            rnd_tensor = np.random.random(size_info)
                            self.ttn_tensor_list[l][i].append(tn.Tensor(rnd_tensor, labels=tensor_label))
                            self.flag_contract[l][i].append(False)
                            self.contracted[l+1][i].append(0)
                else:
                    raise RuntimeError
            elif l == self.num_layers-1:
                if self.layer_type[l] == -1:
                    size_info = (self.bond_inner,)*4+(self.bond_label,)
                    tensor_label = ["D00","D01","D10","D11","U"]
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        self.flag_contract[l].append([])
                        self.contracted[l+1].append([])
                        for j in range(self.layer_size[l]):
                            rnd_tensor = np.random.random(size_info)
                            self.ttn_tensor_list[l][i].append(tn.Tensor(rnd_tensor, labels=tensor_label))
                            self.flag_contract[l][i].append(False)
                            self.contracted[l+1][i].append(0)
                elif self.layer_type[l] == 0:
                    size_info = (self.bond_inner,)*self.channel+(self.bond_label,)
                    tensor_label = ["C"+str(i) for i in range(self.channel)]+["U"]
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        self.flag_contract[l].append([])
                        self.contracted[l+1].append([])
                        for j in range(self.layer_size[l]):
                            rnd_tensor = np.random.random(size_info)
                            self.ttn_tensor_list[l][i].append(tn.Tensor(rnd_tensor, labels=tensor_label))
                            self.flag_contract[l][i].append(False)
                            self.contracted[l+1][i].append(0)
                else:
                    raise RuntimeError
            else:
                if self.layer_type[l] == 1:
                    size_info = (self.bond_inner,)*4 + (self.bond_inner,)
                    tensor_label = ["D00","D01","D10","D11","U"]
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        self.flag_contract[l].append([])
                        self.contracted[l+1].append([])
                        for j in range(self.layer_size[l]):
                            self.ttn_tensor_list[l][i].append([])
                            self.flag_contract[l][i].append([])
                            self.contracted[l+1][i].append([])
                            for c in range(self.channel):
                                rnd_tensor = np.random.random(size_info)
                                self.ttn_tensor_list[l][i][j].append(tn.Tensor(rnd_tensor, labels=tensor_label))
                                self.flag_contract[l][i][j].append(False)
                                self.contracted[l+1][i][j].append(0)
                elif self.layer_type[l] == -1:
                    size_info = (self.bond_inner,)*4+(self.bond_inner,)
                    tensor_label = ["D00","D01","D10","D11","U"]
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        self.flag_contract[l].append([])
                        self.contracted[l+1].append([])
                        for j in range(self.layer_size[l]):
                            rnd_tensor = np.random.random(size_info)
                            self.ttn_tensor_list[l][i].append(tn.Tensor(rnd_tensor, labels=tensor_label))
                            self.flag_contract[l][i].append(False)
                            self.contracted[l+1][i].append(0)
                else:
                    size_info = (self.bond_inner,)*self.channel+(self.bond_inner,)
                    tensor_label = ["C"+str(i) for i in range(self.channel)]+["U"]
                    for i in range(self.layer_size[l]):
                        self.ttn_tensor_list[l].append([])
                        self.flag_contract[l].append([])
                        self.contracted[l+1].append([])
                        for j in range(self.layer_size[l]):
                            rnd_tensor = np.random.random(size_info)
                            self.ttn_tensor_list[l][i].append(tn.Tensor(rnd_tensor, labels=tensor_label))
                            self.flag_contract[l][i].append(False)
                            self.contracted[l+1][i].append(0)
        logger.debug("TTN initialization completed.")

    def get_path_0(self, cl, ci, cj):
        path_len = self.num_layers - cl
        path = [[cl, ci, cj]]
        temp_ci = ci
        temp_cj = cj
        for i in range(1, path_len):
            temp_ci = temp_ci // 2
            temp_cj = temp_cj // 2
            path.append([cl + i, temp_ci, temp_cj])
        return path

    def get_path_1(self, cl, ci, cj, cc):
        path = [[cl, ci, cj, cc]]
        temp_ci = ci
        temp_cj = cj
        for l in range(cl+1,self.num_layers):
            if self.layer_type[l] == 1:
                temp_ci = temp_ci//2
                temp_cj = temp_cj//2
                path.append([l, temp_ci, temp_cj, cc])
            elif self.layer_type[l] == 0:
                path.append([l, temp_ci, temp_cj])
            else:
                temp_ci = temp_ci//2
                temp_cj = temp_cj//2
                path.append([l, temp_ci, temp_cj])
        return path

    def update_single_wo_channel_tensor(self, y, n, cl, ci, cj):
        path = self.get_path_0(cl, ci, cj)

        for l in range(self.num_layers):
            if l == cl:
                for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                    if (not self.flag_contract[l][i][j]) and (i != ci or j != cj):
                        self.contracted[l+1][i][j] = contract_func0(self.ttn_tensor_list[l][i][j],
                                                                    self.contracted[l][2*i][2*j],
                                                                    self.contracted[l][2*i][2*j+1],
                                                                    self.contracted[l][2*i+1][2*j],
                                                                    self.contracted[l][2*i+1][2*j+1],n)
                        self.flag_contract[l][i][j] = True
                        if l < self.num_layers-1:
                            self.flag_contract[l+1][i//2][j//2] = False
                self.contracted[cl+1][ci][cj] = contract_func1(self.contracted[cl][2*ci][2*cj],
                                                               self.contracted[cl][2*ci][2*cj+1],
                                                               self.contracted[cl][2*ci+1][2*cj],
                                                               self.contracted[cl][2*ci+1][2*cj+1],n)
                self.flag_contract[cl][ci][cj] = False
                if cl < self.num_layers-1:
                    self.flag_contract[cl+1][ci//2][cj//2] = False
            elif l > self.layer_comb_channel:
                for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                    if not self.flag_contract[l][i][j]:
                        if ([l,i,j] in path) and (l-1==cl):
                            if ci%2==0 and cj%2==0:
                                contracted_list = [self.contracted[l][ci][cj+1],self.contracted[l][ci+1][cj],self.contracted[l][ci+1][cj+1]]
                                edge_list = ['D01', 'D10', 'D11']
                            elif ci%2==0 and cj%2==1:
                                contracted_list = [self.contracted[l][ci][cj-1],self.contracted[l][ci+1][cj-1],self.contracted[l][ci+1][cj]]
                                edge_list = ['D00', 'D10', 'D11']
                            elif ci%2==1 and cj%2==0:
                                contracted_list = [self.contracted[l][ci-1][cj],self.contracted[l][ci-1][cj+1],self.contracted[l][ci][cj+1]]
                                edge_list = ['D00', 'D01', 'D11']
                            else:
                                contracted_list = [self.contracted[l][ci-1][cj-1],self.contracted[l][ci-1][cj],self.contracted[l][ci][cj-1]]
                                edge_list = ['D00', 'D01', 'D10']
                            self.contracted[l+1][i][j] = contract_func2(self.ttn_tensor_list[l][i][j], contracted_list, edge_list, n)
                            self.flag_contract[l][i][j] = False
                            if l < self.num_layers-1:
                                self.flag_contract[l+1][i//2][j//2] = False
                        else:
                            self.contracted[l+1][i][j] = contract_func0(self.ttn_tensor_list[l][i][j],
                                                                        self.contracted[l][2*i][2*j],
                                                                        self.contracted[l][2*i][2*j+1],
                                                                        self.contracted[l][2*i+1][2*j],
                                                                        self.contracted[l][2*i+1][2*j+1],n)
                            self.flag_contract[l][i][j] = (not ([l,i,j] in path))
                            if l < self.num_layers-1:
                                self.flag_contract[l+1][i//2][j//2] = False

            elif l == self.layer_comb_channel:
                for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                    if not self.flag_contract[l][i][j]:
                        self.contracted[l+1][i][j] = contract_func3(self.ttn_tensor_list[l][i][j], self.contracted[l][i][j], n)
                        self.flag_contract[l][i][j] = True
                        if l < self.num_layers-1:
                            self.flag_contract[l+1][i//2][j//2] = False
                        
        if cl != self.num_layers-1:
            tempD = tn.zeros_tensor([n, self.bond_inner], labels=['S', 'B'])
            for s, b in product(range(n), range(self.bond_inner)):
                sum1 = 0
                for f in range(self.bond_label):
                    sum1 = sum1 + self.contracted[self.num_layers][0][0].data[s,f,b] * y.data[s,f]
                tempD.data[s, b] = sum1
            tensor_environment = tn.contract(self.contracted[cl+1][ci][cj], tempD, ["S"], ["S"])
        else:
            tensor_environment = tn.contract(self.contracted[self.num_layers][0][0], y, "S", "S")
        bond_out = self.bond_label if cl == self.num_layers-1 else self.bond_inner
        matrix = np.reshape(tensor_environment.data,
                (self.bond_inner*self.bond_inner*self.bond_inner*self.bond_inner, bond_out))
        u, sigma, vt = np.linalg.svd(matrix, 0)
        self.ttn_tensor_list[cl][ci][cj].data = np.reshape(
                np.dot(u, vt), (self.bond_inner, self.bond_inner, self.bond_inner, self.bond_inner, bond_out))

    def update_single_comb_channel_tensor(self, y, n, cl, ci, cj):
        path = self.get_path_0(cl, ci, cj)
        for l in range(self.num_layers):
            if l == cl:
                for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                    if (not self.flag_contract[l][i][j]) and (i != ci or j != cj):
                        self.contracted[l+1][i][j] = contract_func3(self.ttn_tensor_list[l][i][j],
                                                                    self.contracted[l][i][j], n)
                        self.flag_contract[l][i][j] = True
                        if l < self.num_layers-1:
                            self.flag_contract[l+1][i//2][j//2] = False
                self.contracted[cl+1][ci][cj] = contract_func4(self.contracted[cl][ci][cj], n)
                self.flag_contract[cl][ci][cj] = False
                if cl < self.num_layers-1:
                    self.flag_contract[cl+1][ci//2][cj//2] = False
            elif l < cl:
                for i, j, c in product(range(self.layer_size[l]), range(self.layer_size[l]), range(self.channel)):
                    if not self.flag_contract[l][i][j][c]:
                        self.contracted[l+1][i][j][c] = contract_func0(self.ttn_tensor_list[l][i][j][c],
                                                                       self.contracted[l][2*i][2*j][c],
                                                                       self.contracted[l][2*i][2*j+1][c],
                                                                       self.contracted[l][2*i+1][2*j][c],
                                                                       self.contracted[l][2*i+1][2*j+1][c], n)
                        self.flag_contract[l][i][j][c] = True
                        if l == self.layer_comb_channel-1:
                            self.flag_contract[l+1][i][j] = False
                        else:
                            self.flag_contract[l+1][i//2][j//2][c] = False
            else:
                for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                    if not self.flag_contract[l][i][j]:
                        if ([l,i,j] in path) and (l-1==cl):
                            if ci%2==0 and cj%2==0:
                                contracted_list = [self.contracted[l][ci][cj+1],self.contracted[l][ci+1][cj],self.contracted[l][ci+1][cj+1]]
                                edge_list = ['D01', 'D10', 'D11']
                            elif ci%2==0 and cj%2==1:
                                contracted_list = [self.contracted[l][ci][cj-1],self.contracted[l][ci+1][cj-1],self.contracted[l][ci+1][cj]]
                                edge_list = ['D00', 'D10', 'D11']
                            elif ci%2==1 and cj%2==0:
                                contracted_list = [self.contracted[l][ci-1][cj],self.contracted[l][ci-1][cj+1],self.contracted[l][ci][cj+1]]
                                edge_list = ['D00', 'D01', 'D11']
                            else:
                                contracted_list = [self.contracted[l][ci-1][cj-1],self.contracted[l][ci-1][cj],self.contracted[l][ci][cj-1]]
                                edge_list = ['D00', 'D01', 'D10']
                            self.contracted[l+1][i][j] = contract_func2(self.ttn_tensor_list[l][i][j], contracted_list, edge_list, n)
                            self.flag_contract[l][i][j] = False
                            if l < self.num_layers-1:
                                self.flag_contract[l+1][i//2][j//2] = False
                        else:
                            self.contracted[l+1][i][j] = contract_func0(self.ttn_tensor_list[l][i][j],
                                                                        self.contracted[l][2*i][2*j],
                                                                        self.contracted[l][2*i][2*j+1],
                                                                        self.contracted[l][2*i+1][2*j],
                                                                        self.contracted[l][2*i+1][2*j+1],n)
                            self.flag_contract[l][i][j] = (not ([l,i,j] in path))
                            if l < self.num_layers-1:
                                self.flag_contract[l+1][i//2][j//2] = False

        if cl != self.num_layers-1:
            tempD = tn.zeros_tensor([n, self.bond_inner], labels=['S', 'B'])
            for s, b in product(range(n), range(self.bond_inner)):
                sum1 = 0
                for f in range(self.bond_label):
                    sum1 = sum1 + self.contracted[self.num_layers][0][0].data[s,f,b] * y.data[s,f]
                tempD.data[s, b] = sum1
            tensor_environment = tn.contract(self.contracted[cl+1][ci][cj], tempD, ["S"], ["S"])
        else:
            tensor_environment = tn.contract(self.contracted[self.num_layers][0][0], y, "S", "S")
        bond_in = self.bond_data if cl == 0 else self.bond_inner
        bond_out = self.bond_label if cl == self.num_layers-1 else self.bond_inner
        matrix = np.reshape(tensor_environment.data,
                (bond_in**self.channel, bond_out))
        u, sigma, vt = np.linalg.svd(matrix, 0)
        self.ttn_tensor_list[cl][ci][cj].data = np.reshape(
                np.dot(u, vt), (bond_in,)*self.channel+(bond_out,))


    def update_single_w_channel_tensor(self, y, n, cl, ci, cj, cc):
        path = self.get_path_1(cl, ci, cj, cc)
        for l in range(self.num_layers):
            if cl == l:
                for i, j, c in product(range(self.layer_size[l]), range(self.layer_size[l]), range(self.channel)):
                    if (not self.flag_contract[l][i][j][c]) and (i != ci or j != cj or c != cc):
                        self.contracted[l+1][i][j][c] = contract_func0(self.ttn_tensor_list[l][i][j][c],
                                                                       self.contracted[l][2*i][2*j][c],
                                                                       self.contracted[l][2*i][2*j+1][c],
                                                                       self.contracted[l][2*i+1][2*j][c],
                                                                       self.contracted[l][2*i+1][2*j+1][c], n)
                        self.flag_contract[l][i][j][c] = True
                        if l == self.layer_comb_channel-1:
                            self.flag_contract[l+1][i][j] = False
                        else:
                            self.flag_contract[l+1][i//2][j//2][c] = False
                self.contracted[cl+1][ci][cj][cc] = contract_func1(self.contracted[cl][2*ci][2*cj][cc],
                                                                   self.contracted[cl][2*ci][2*cj+1][cc],
                                                                   self.contracted[cl][2*ci+1][2*cj][cc],
                                                                   self.contracted[cl][2*ci+1][2*cj+1][cc],n)
                self.flag_contract[cl][ci][cj][cc] = False
                if cl == self.layer_comb_channel-1:
                    self.flag_contract[cl+1][ci][cj] = False
                else:
                    self.flag_contract[cl+1][ci//2][cj//2][cc] = False
            elif l < self.layer_comb_channel:
                for i, j, c in product(range(self.layer_size[l]), range(self.layer_size[l]), range(self.channel)):
                    if not self.flag_contract[l][i][j][c]:
                        if([l,i,j,c] in path) and (l-1==cl):
                            if ci%2==0 and cj%2==0:
                                contracted_list = [self.contracted[l][ci][cj+1][c],self.contracted[l][ci+1][cj][c],self.contracted[l][ci+1][cj+1][c]]
                                edge_list = ['D01', 'D10', 'D11']
                            elif ci%2==0 and cj%2==1:
                                contracted_list = [self.contracted[l][ci][cj-1][c],self.contracted[l][ci+1][cj-1][c],self.contracted[l][ci+1][cj][c]]
                                edge_list = ['D00', 'D10', 'D11']
                            elif ci%2==1 and cj%2==0:
                                contracted_list = [self.contracted[l][ci-1][cj][c],self.contracted[l][ci-1][cj+1][c],self.contracted[l][ci][cj+1][c]]
                                edge_list = ['D00', 'D01', 'D11']
                            else:
                                contracted_list = [self.contracted[l][ci-1][cj-1][c],self.contracted[l][ci-1][cj][c],self.contracted[l][ci][cj-1][c]]
                                edge_list = ['D00', 'D01', 'D10']
                            self.contracted[l+1][i][j][c] = contract_func2(self.ttn_tensor_list[l][i][j][c], contracted_list, edge_list, n)
                            self.flag_contract[l][i][j][c] = False
                            if l == self.layer_comb_channel-1:
                                self.flag_contract[l+1][i][j] = False
                            else:
                                self.flag_contract[l+1][i//2][j//2][c] = False
                        else:
                            self.contracted[l+1][i][j][c] = contract_func0(self.ttn_tensor_list[l][i][j][c],
                                                                           self.contracted[l][2*i][2*j][c],
                                                                           self.contracted[l][2*i][2*j+1][c],
                                                                           self.contracted[l][2*i+1][2*j][c],
                                                                           self.contracted[l][2*i+1][2*j+1][c], n)
                            self.flag_contract[l][i][j][c] = (not ([l,i,j,c] in path))
                            if l == self.layer_comb_channel-1:
                                self.flag_contract[l+1][i][j] = False
                            else:
                                self.flag_contract[l+1][i//2][j//2][c] = False
            elif l == self.layer_comb_channel:
                for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                    if not self.flag_contract[l][i][j]:
                        if([l,i,j] in path) and (l-1==cl):
                            contracted_list = copy.deepcopy(self.contracted[l][i][j])
                            edge_list = ['C'+str(i) for i in range(self.channel)]
                            contracted_list.pop(cc)
                            edge_list.pop(cc)
                            self.contracted[l+1][i][j] = contract_func2(self.ttn_tensor_list[l][i][j], contracted_list, edge_list, n)
                            self.flag_contract[l][i][j] = False
                            if l < self.num_layers-1:
                                self.flag_contract[l+1][i//2][j//2] = False
                        else:
                            self.contracted[l+1][i][j] = contract_func3(self.ttn_tensor_list[l][i][j],
                                                                        self.contracted[l][i][j], n)
                            self.flag_contract[l][i][j] = (not ([l,i,j] in path))
                            if l < self.num_layers-1:
                                self.flag_contract[l+1][i//2][j//2] = False
            elif l > self.layer_comb_channel:
                for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                    if not self.flag_contract[l][i][j]:
                        self.contracted[l+1][i][j] = contract_func0(self.ttn_tensor_list[l][i][j],
                                                                    self.contracted[l][2*i][2*j],
                                                                    self.contracted[l][2*i][2*j+1],
                                                                    self.contracted[l][2*i+1][2*j],
                                                                    self.contracted[l][2*i+1][2*j+1], n)
                        self.flag_contract[l][i][j] = (not ([l,i,j] in path))
                        if l < self.num_layers-1:
                            self.flag_contract[l+1][i//2][j//2] = False

        tempD = tn.zeros_tensor([n, self.bond_inner], labels=['S', 'B'])
        for s, b in product(range(n), range(self.bond_inner)):
            sum1 = 0
            for f in range(self.bond_label):
                sum1 = sum1 + self.contracted[self.num_layers][0][0].data[s,f,b] * y.data[s,f]
            tempD.data[s, b] = sum1

        tensor_environment = tn.contract(self.contracted[cl+1][ci][cj][cc], tempD, ["S"], ["S"])

        bond_in = self.bond_data if cl == 0 else self.bond_inner
        matrix = np.reshape(tensor_environment.data,
                (bond_in*bond_in*bond_in*bond_in, self.bond_inner))
        u, sigma, vt = np.linalg.svd(matrix, 0)
        self.ttn_tensor_list[cl][ci][cj][cc].data = np.reshape(
                np.dot(u, vt), (bond_in,bond_in,bond_in,bond_in,self.bond_inner))

    def train(self, x, y, epoch):
        n_train = y.shape[0]
        data_tensor_list = []
        for i in range(self.size):
            data_tensor_list.append([])
            for j in range(self.size):
                data_tensor_list[i].append([])
                for c in range(self.channel):
                    data_tensor_list[i][j].append(tn.Tensor(x[:,i,j,c,:],labels=["S","U"]))
        self.contracted[0] = data_tensor_list

        label_tensor = tn.Tensor(y,labels=["S","U"])

        for n in range(epoch):
            for l in range(self.num_layers):
                if self.layer_type[l] == -1:
                    for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                        self.update_single_wo_channel_tensor(label_tensor, n_train, l, i, j)
                        logger.debug(f"Done one wo tensor ({l},{i},{j}).")
                elif self.layer_type[l] == 0:
                    for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                        self.update_single_comb_channel_tensor(label_tensor, n_train, l, i, j)
                        logger.debug(f"Done one comb tensor ({l},{i},{j}).")
                else:
                    for i, j, c in product(range(self.layer_size[l]), range(self.layer_size[l]), range(self.channel)):
                        self.update_single_w_channel_tensor(label_tensor, n_train, l, i, j, c)
                        logger.debug(f"Done one w tensor ({l},{i},{j},{c}).")
            logger.debug(f'Epoch {n} completed.')

    def predict(self, x):
        n_sample = x.shape[0]
        data_tensor_list = []
        for i in range(self.size):
            data_tensor_list.append([])
            for j in range(self.size):
                data_tensor_list[i].append([])
                for c in range(self.channel):
                    data_tensor_list[i][j].append(tn.Tensor(x[:,i,j,c,:],labels=["S","U"]))
        self.contracted[0] = data_tensor_list

        for l in range(self.num_layers):
            if self.layer_type[l] == 1:
                for i, j, c in product(range(self.layer_size[l]), range(self.layer_size[l]), range(self.channel)):
                    self.contracted[l+1][i][j][c] = contract_func0(self.ttn_tensor_list[l][i][j][c],
                                                                   self.contracted[l][2*i][2*j][c],
                                                                   self.contracted[l][2*i][2*j+1][c],
                                                                   self.contracted[l][2*i+1][2*j][c],
                                                                   self.contracted[l][2*i+1][2*j+1][c], n_sample)
            elif self.layer_type[l] == 0:
                for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                    self.contracted[l+1][i][j] = contract_func3(self.ttn_tensor_list[l][i][j], self.contracted[l][i][j], n_sample)
            else:
                for i, j in product(range(self.layer_size[l]), range(self.layer_size[l])):
                    self.contracted[l+1][i][j] = contract_func0(self.ttn_tensor_list[l][i][j], 
                                                                self.contracted[l][2*i][2*j],
                                                                self.contracted[l][2*i][2*j+1],
                                                                self.contracted[l][2*i+1][2*j],
                                                                self.contracted[l][2*i+1][2*j+1], n_sample)
        return self.contracted[self.num_layers][0][0].data
