def read_image_dataset(dataset, path, kind):
    if dataset == "MNIST":
        return load_mnist(path, kind)
    elif dataset == "Fashion-MNIST":
        return load_fashion(path, kind)
    elif dataset == "CIFAR10":
        return load_cifar10(path, kind)
    else:
        raise NotImplementedError

def load_mnist(path, kind):
    import os
    import gzip
    import numpy as np

    if kind == "train":
        labels_path = os.path.join(path, 'MNIST/train-labels-idx1-ubyte.gz')
        images_path = os.path.join(path, 'MNIST/train-images-idx3-ubyte.gz')
    elif kind == "test":
        labels_path = os.path.join(path, 'MNIST/t10k-labels-idx1-ubyte.gz')
        images_path = os.path.join(path, 'MNIST/t10k-images-idx3-ubyte.gz')
    else:
        raise NotImplementedError

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    label_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    images = np.reshape(images, (len(labels), 1, 28, 28))

    return images, labels, label_names

def load_fashion(path, kind):
    import os
    import gzip
    import numpy as np

    if kind == "train":
        labels_path = os.path.join(path, 'Fashion/train-labels-idx1-ubyte.gz')
        images_path = os.path.join(path, 'Fashion/train-images-idx3-ubyte.gz')
    elif kind == "test":
        labels_path = os.path.join(path, 'Fashion/t10k-labels-idx1-ubyte.gz')
        images_path = os.path.join(path, 'Fashion/t10k-images-idx3-ubyte.gz')
    else:
        raise NotImplementedError

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    images = np.reshape(images, (len(labels), 1, 28, 28))

    return images, labels, label_names

def load_cifar10(path, kind='train'):
    import os
    import numpy as np

    def unpickle(f):
        import pickle
        with open(f, 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
        return dic

    if kind == "train":
        images = np.empty(shape=(0,3072))
        llabels = []
        for i in range(1,6):
            data_file = os.path.join(path,"CIFAR10/data_batch_%d" % i)
            d = unpickle(data_file)
            images_i = d[b"data"]
            labels_i = d[b"labels"]
            images = np.concatenate((images, images_i))
            llabels += labels_i
    elif kind == "test":
        data_file = os.path.join(path,"CIFAR10/test_batch")
        d = unpickle(data_file)
        images = d[b"data"]
        llabels = d[b"labels"]
    else:
        raise NotImplementedError
    labels = np.array(llabels, dtype=np.uint8)
    images = images.astype(np.uint8)

    f_label_name = os.path.join(path,"CIFAR10/batches.meta")
    d = unpickle(f_label_name)
    label_names = d[b"label_names"]

    images = np.reshape(images, (len(labels), 3, 32, 32))

    return images, labels, label_names
