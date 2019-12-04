import logging
logger = logging.getLogger('qmle')

def image_subsample(image):
    import numpy as np

    size = image.shape[1]
    size_extend = 2**int(np.ceil(np.log2(size)))
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
    return image_shrink

def image_normalize(image):
    return image/256

def get_image_group(images, labels):
    import numpy as np
    n = np.min([np.sum(labels==i) for i in range(10)])
    image_group = np.zeros((16, 16, 10, n), dtype=np.float64)
    dict_id = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    for image, label in zip(images, labels):
        if dict_id[label] >= n:
            continue
        image_group[:,:,label,dict_id[label]] = np.squeeze(image_normalize(image_subsample(image)))
        dict_id[label] += 1
    return image_group

def preprocess(images, labels, n_train, n_train_each, bond_label, bond_data, current_class):
    import numpy as np
    import tncontract as tn
    from itertools import product
    from itertools import combinations

    image_group = get_image_group(images, labels)

    data_group = np.zeros(
        (16, 16, 10, image_group.shape[3], bond_data), dtype=np.float64)

    for i in range(bond_data):
        data_group[:, :, :, :, i] = (len([c for c in combinations(range(bond_data - 1), i)]) ** 0.5) * \
            np.cos((image_group) * (np.pi / 2)) ** (bond_data - (i + 1)) * np.sin(
            (image_group) * (np.pi / 2)) ** i

    train_data = np.zeros((16, 16, bond_data, n_train), dtype=np.float64)
    label_data = np.zeros((n_train, bond_label), dtype=np.float64)

    for k, m in product(range(bond_data), range(n_train_each)):
        train_data[:, :, k, m] = data_group[:, :, current_class, m, k]

    cc = set([current_class])
    rest = set(range(0, 10)) - cc

    for k, l, m in product(range(bond_data), range(9), range(int(n_train_each / 9))):
        train_data[:, :, k, n_train_each + l *
                   int(n_train_each / 9) + m] = data_group[:, :, list(rest)[l], m, k]

    label_data[0:n_train_each] = [1, 0]
    label_data[n_train_each:n_train] = [0, 1]

    data_tensor = [[0 for col in range(16)] for row in range(16)]
    for i, j in product(range(16), range(16)):
        data_tensor[i][j] = tn.Tensor(
            train_data[i, j, :, :], labels=["up", "down"])

    label_tensor = tn.Tensor(label_data, labels=["up", "down"])
    return data_tensor, label_tensor

def preprocess_test(images, labels, n_test, n_test_each, bond_data):
    import numpy as np
    import tncontract as tn
    from itertools import product
    from itertools import combinations

    image_group = get_image_group(images, labels)

    data_group = np.zeros(
        (16, 16, 10, image_group.shape[3], bond_data), dtype=np.float64)

    for i in range(bond_data):
        data_group[:, :, :, :, i] = (len([c for c in combinations(range(bond_data - 1), i)]) ** 0.5) * \
            np.cos((image_group) * (np.pi / 2)) ** (bond_data - (i + 1)) * np.sin(
            (image_group) * (np.pi / 2)) ** i

    test_data = np.zeros((16, 16, bond_data, n_test), dtype=np.float64)

    for k, l, m in product(range(bond_data), range(10), range(n_test_each)):
        test_data[:, :, k, l * n_test_each + m] = data_group[:, :, l, m, k]

    test_tensor = [[0 for col in range(16)] for row in range(16)]
    for i, j in product(range(16), range(16)):
        test_tensor[i][j] = tn.Tensor(
            test_data[i, j, :, :], labels=["up", "down"])

    return test_tensor

def run_ttn_ref(args, x_train, l_train, x_test, l_test):
    from itertools import product
    import numpy as np
    import pickle

    from .tree_tensor_network_mnist import TreeTensorNetwork

    data_folder = "./data/mnist/"
    n_epochs = 3

    bond_data = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    bond_inner = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    bond_label = 2

    n_class = 2
    n_train_single = 10
    n_train_each = n_train_single * 9
    n_train = n_train_each * n_class

    n_test_each = 800
    n_test = n_test_each * 10

    layer_units = [16, 8, 4, 2, 1]


    # build tensor network---------------------------------------------------
    logger.info("building tensor network")
    output = open('10_class_model.pkl', 'wb')

    ttn = [0 for col in range(10)]
    acc_train = [0 for col in range(10)]
    acc_test1 = [0 for col in range(10)]
    acc_test2 = [0 for col in range(10)]

    for i in range(10):
        logger.info(f"bond_data: {bond_data[i]}")
        logger.info(f"bond_inner: {bond_inner[i]}")
        data, labels = preprocess(x_train, l_train, n_train, n_train_each, bond_label, bond_data[i], i)
        data_test, labels_test = preprocess(x_test, l_test, n_test_each*2, n_test_each, bond_label, bond_data[i], i)

        ttn[i] = TreeTensorNetwork(
            data, labels, bond_data[i], bond_inner[i], bond_label, layer_units)

        # Training
        logger.info(f"Training number {i}-th binary classifier:")
        acc_train[i] = ttn[i].train(n_epochs)
        logger.info(f"training inner product: {acc_train[i]}")

        # Testing for each classifier
        logger.info(f"Testing number {i}-th binary classifier:")
        acc_test1[i], acc_test2[i] = ttn[i].test(data_test, labels_test)
        logger.info(f"testing accuracy {acc_test2[i]}")

        pickle.dump(ttn, output)
    output.close()

    # Testing on 10 classes
    output_vector = np.zeros((n_test, 10), dtype=np.float64)
    output_label = np.zeros((n_test), dtype=np.float64)
    count = 0
    for i in range(10):
        test_tensor = preprocess_test(x_test, l_test, n_test, n_test_each, bond_data[i])
        output_vector[:, i] = ttn[i].outputvalue(test_tensor, n_test)

    for i, j in product(range(10), range(n_test_each)):
        output_label[i * n_test_each +
                     j] = np.argmax(output_vector[i * n_test_each + j, :])
        if output_label[i * n_test_each + j] == i:
            count = count + 1

    logger.info(f"testing accuracy on 10 classes: {count/n_test}")
