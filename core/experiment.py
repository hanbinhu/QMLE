import logging
logger = logging.getLogger("qmle")

def run_core(args, images_train, labels_train, images_test, labels_test, label_names):
    from utils.plot_func import plot_images, pltshow
    plot_images(images_train)
    pltshow()
