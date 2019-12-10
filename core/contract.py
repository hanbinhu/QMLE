import numpy as np
import tncontract as tn
from itertools import product
from sklearn import preprocessing

def contract_func0(tensor0, tensor1, tensor2, tensor3, tensor4, n):
    temp = contract_func1(tensor1, tensor2, tensor3, tensor4, n)
    tensor_result = tn.contract(temp, tensor0, ["D00", "D01", "D10", "D11"], ["D00", "D01", "D10", "D11"])

    if len(tensor_result.shape) == 2:
        tensor_result.data = preprocessing.normalize(tensor_result.data, axis=0, norm='l2')
    else:
        for i in range(tensor_result.shape[2]):
            tensor_result.data[:,:,i] = preprocessing.normalize(tensor_result.data[:,:,i], axis=0, norm='l2')
    return tensor_result

def contract_func1(tensor1, tensor2, tensor3, tensor4, n):
    bond = tensor1.shape[1]
    if len(tensor1.shape) == 2 and len(tensor2.shape) == 2 and len(tensor3.shape) == 2 and len(tensor4.shape) == 2:
        tensor_result = tn.random_tensor(n, bond, bond, bond, bond, labels=['S','D00', 'D01', 'D10', 'D11'])
        for i, j, k, l in product(range(bond), range(bond), range(bond), range(bond)):
            tensor_result.data[:, i, j, k, l] = tensor1.data[:,i]*tensor2.data[:,j]*tensor3.data[:,k]*tensor4.data[:,l]
    else:
        if len(tensor1.shape) == 3:
            bond_inner = tensor1.shape[2]
            tensor_result = tn.random_tensor(n, bond, bond, bond, bond, bond_inner, labels=['S', 'D00', 'D01', 'D10', 'D11', 'U'])
            for i, j, k, l, m in product(range(bond), range(bond), range(bond), range(bond), range(bond_inner)):
                tensor_result.data[:,i,j,k,l,m] = tensor1.data[:,i,m]*tensor2.data[:,j]*tensor3.data[:,k]*tensor4.data[:,l]
        if len(tensor2.shape) == 3:
            bond_inner = tensor2.shape[2]
            tensor_result = tn.random_tensor(n, bond, bond, bond, bond, bond_inner, labels=['S', 'D00', 'D01', 'D10', 'D11', 'U'])
            for i, j, k, l, m in product(range(bond), range(bond), range(bond), range(bond), range(bond_inner)):
                tensor_result.data[:,i,j,k,l,m] = tensor1.data[:,i]*tensor2.data[:,j,m]*tensor3.data[:,k]*tensor4.data[:,l]
        if len(tensor3.shape) == 3:
            bond_inner = tensor3.shape[2]
            tensor_result = tn.random_tensor(n, bond, bond, bond, bond, bond_inner, labels=['S', 'D00', 'D01', 'D10', 'D11', 'U'])
            for i, j, k, l, m in product(range(bond), range(bond), range(bond), range(bond), range(bond_inner)):
                tensor_result.data[:,i,j,k,l,m] = tensor1.data[:,i]*tensor2.data[:,j]*tensor3.data[:,k,m]*tensor4.data[:,l]
        if len(tensor4.shape) == 3:
            bond_inner = tensor4.shape[2]
            tensor_result = tn.random_tensor(n, bond, bond, bond, bond, bond_inner, labels=['S', 'D00', 'D01', 'D10', 'D11', 'U'])
            for i, j, k, l, m in product(range(bond), range(bond), range(bond), range(bond), range(bond_inner)):
                tensor_result.data[:,i,j,k,l,m] = tensor1.data[:,i]*tensor2.data[:,j]*tensor3.data[:,k]*tensor4.data[:,l,m]
    return tensor_result

def contract_local(tensor_list, n):
    ch = max(1,len(tensor_list))
    if len(tensor_list) == 0:
        bond = 1
    else:
        bond = tensor_list[0].shape[1]
    size_list = [n]+[bond]*ch
    edge_list = ['C'+str(i) for i in range(ch)]
    tensor_result = tn.random_tensor(*size_list, labels=['S']+edge_list)
    for index in product(*([range(bond)]*ch)):
        nindex = (slice(0,n),)+index
        temp = np.ones((n,))
        if len(tensor_list) > 0:
            for t, i in enumerate(index):
                temp *= tensor_list[t].data[:,i]
        tensor_result.data[nindex] = temp
    return tensor_result, edge_list

def contract_func2(tensor0, tensor_list, edge_list, n):
    temp, temp_edge_list = contract_local(tensor_list, n)
    tensor_result = tn.contract(temp, tensor0, temp_edge_list, edge_list)
    tensor_result.data = tensor_result.data.transpose(0, 2, 1)
    tensor_result.labels[1], tensor_result.labels[2] = tensor_result.labels[2], tensor_result.labels[1]
    for i in range(tensor_result.shape[2]):
        tensor_result.data[:,:,i] = preprocessing.normalize(tensor_result.data[:,:,i], axis=0, norm='l2')
    return tensor_result

def contract_func3(tensor0, tensor_list, n):
    channel = len(tensor_list)
    elist = ['C'+str(i) for i in range(channel)]
    temp = contract_func4(tensor_list, n)

    tensor_result = tn.contract(temp, tensor0, elist, elist)

    if len(tensor_result.shape) == 2:
        tensor_result.data = preprocessing.normalize(tensor_result.data, axis=0, norm='l2')
    else:
        for i in range(tensor_result.shape[2]):
            tensor_result.data[:,:,i] = preprocessing.normalize(tensor_result.data[:,:,i], axis=0, norm='l2')
    return tensor_result

def contract_func4(tensor_list, n):
    channel = len(tensor_list)
    bond = tensor_list[0].shape[1]
    if all([len(tensor.shape)==2 for tensor in tensor_list]):
        size_list=[n]+[bond]*channel
        tensor_result = tn.random_tensor(*size_list, labels=['S']+['C'+str(i) for i in range(channel)])
        for index in product(*([range(bond)]*channel)):
            temp = np.ones((n,))
            for t, i in enumerate(index):
                temp *= tensor_list[t].data[:,i]
            nindex = (slice(0,n),)+index
            tensor_result.data[nindex] = temp
    else:
        for c in range(channel):
            if len(tensor_list[c].shape) != 2:
                bond_inner = tensor_list[c].shape[2]
                size_list = [n]+[bond]*channel+[bond_inner]
                tensor_result = tn.random_tensor(*size_list, labels=['S']+['C'+str(i) for i in range(channel)]+['U'])
                for index in product(*([range(bond)]*channel+[range(bond_inner)])):
                    temp = np.ones((n,))
                    for t, i in enumerate(index):
                        if t != c:
                            temp *= tensor_list[t].data[:,i]
                        else:
                            temp *= tensor_list[t].data[:,i,index[-1]]
                    nindex = (slice(0,n),)+index
                    tensor_result.data[nindex] = temp
                break
    return tensor_result
