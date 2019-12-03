import argparse
from utils.read_data import read_image_dataset
from utils.plot_func import plot_images, pltshow

def main():
    images, labels, label_names = read_image_dataset('MNIST', './data', 'train')
    plot_images(images)
    images, labels, label_names = read_image_dataset('Fashion-MNIST', './data', 'train')
    plot_images(images)
    images, labels, label_names = read_image_dataset('CIFAR10', './data', 'train')
    plot_images(images)
    pltshow()

if __name__ == '__main__':
    main()
