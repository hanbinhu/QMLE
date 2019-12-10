import logging
logger = logging.getLogger('qmle')

def image_subsample(image):
    import numpy as np

    size = image.shape[1]
    size_extend = 2**int(np.ceil(np.log2(size)))
    if size != size_extend:
        image_extend = np.zeros((image.shape[0], size_extend, size_extend), dtype=np.float64)
        l = int(np.floor((size_extend-size)/2))
        r = l+size
        image_extend[:,l:r,l:r] = image
        image_extend[:,:l,:l] = image_extend[:,l,l]
        image_extend[:,:l,r:] = image_extend[:,l,r-1]
        image_extend[:,r:,:l] = image_extend[:,r-1,l]
        image_extend[:,r:,r:] = image_extend[:,r-1,r-1]

        image_extend[:,:l,l:r] = image_extend[:,l,l:r]
        image_extend[:,r:,l:r] = image_extend[:,r-1,l:r]
        image_extend[:,l:r,:l] = image_extend[:,l:r,l:l+1]
        image_extend[:,l:r,r:] = image_extend[:,l:r,r-1:r]

        image_shrink = (image_extend[:,::2,::2]+image_extend[:,1::2,::2]+image_extend[:,::2,1::2]+image_extend[:,1::2,1::2])/4
    else:
        image_shrink = image
    return image_shrink

def image_normalize(image):
    #TODO: Better Normalization Method
    return image/256

def get_image_group(images, labels, n_class):
    import numpy as np
    channel = images.shape[1]

    size = images.shape[2]
    size_extend = 2**int(np.ceil(np.log2(size)))
    if size_extend != size:
        size = size_extend // 2

    n = np.min([np.sum(labels==i) for i in range(n_class)])
    image_group = np.zeros((size, size, channel, n_class, n), dtype=np.float64)
    dict_id = {i:0 for i in range(n_class)}
    for image, label in zip(images, labels):
        if dict_id[label] >= n:
            continue
        image_group[:,:,:,label,dict_id[label]] = image_normalize(image_subsample(image)).transpose(1,2,0)
        dict_id[label] += 1
    return image_group

def get_data_group(image_group, bond_data):
    import numpy as np
    from itertools import combinations
    data_group_shape = image_group.shape+(bond_data,)
    data_group = np.zeros(data_group_shape, dtype=np.float64)

    for i in range(bond_data):
        data_group[:, :, :, :, :, i] = (len(list(combinations(range(bond_data - 1), i))) ** 0.5) * \
            np.cos((image_group) * (np.pi / 2)) ** (bond_data - (i + 1)) * np.sin(
            (image_group) * (np.pi / 2)) ** i
    return data_group

def preprocess_data_ova(data_group, n_sample):
    import numpy as np
    from itertools import product
    size, _, channel, n_class, n_tot, bond_data = data_group.shape
    x = np.zeros((size, size, channel, bond_data, n_sample), dtype=np.float64)
    y = np.zeros((n_sample, 2), dtype=np.float64)

    n_each = n_sample // n_class
    n_each_one = n_each // 2
    n_each_sub = n_each_one // (n_class-1)
    for current_class in range(n_class):
        s = current_class*n_each
        for k, m in product(range(bond_data), range(n_each_one)):
            x[:, :, :, k, s+m] = data_group[:, :, :, current_class, m, k]
            y[s+m, 0] = 1.0

        cc = set([current_class])
        rest = set(range(n_class)) - cc

        for k, l, m in product(range(bond_data), range(n_class-1), range(int(n_each_sub))):
            x[:, :, :, k, s+n_each_one+l*n_each_sub+m] = data_group[:, :, :, list(rest)[l], m, k]
            y[s+n_each_one+l*n_each_sub+m, 1] = 1.0

    return x, y

def preprocess_data_oh(data_group, n_sample):
    import numpy as np
    from itertools import product
    size, _, channel, n_class, n_tot, bond_data = data_group.shape
    x = np.zeros((size, size, channel, bond_data, n_sample), dtype=np.float64)
    y = np.zeros((n_sample, n_class), dtype=np.float64)
    n_each = n_sample // n_class

    for current_class in range(n_class):
        s = current_class*n_each
        for k, m in product(range(bond_data), range(n_each)):
            x[:, :, :, k, s+m] = data_group[:, :, :, current_class, m, k]
            y[s+m, current_class] = 1.0

    return x, y

def preprocess_data(ctype, images, labels, n_class, n_sample, bond_data):
    image_group = get_image_group(images, labels, n_class)
    data_group = get_data_group(image_group, bond_data)
    if ctype == 'one-vs-all':
        assert(n_sample%(n_class*(n_class-1)*2) == 0)
        x, y = preprocess_data_ova(data_group, n_sample)
    elif ctype == 'one-hot':
        assert(n_sample%n_class == 0)
        x, y = preprocess_data_oh(data_group, n_sample)
    else:
        raise NotImplementedError
    return x, y
