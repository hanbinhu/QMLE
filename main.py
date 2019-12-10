#!/usr/bin/python3

def main():
    from utils.read_data import read_image_dataset
    from utils.arg_parse import get_args
    from third_party.ttn_ref import run_ttn_ref
    from core.experiment import run_core

    args = get_args()
    if args.command == 'reproduce':
        images_train, labels_train, _ = read_image_dataset('MNIST', args.data_path, 'train')
        images_test, labels_test, _ = read_image_dataset('MNIST', args.data_path, 'test')
        run_ttn_ref(args, images_train, labels_train, images_test, labels_test)
    elif args.command == 'run':
        images_train, labels_train, _ = read_image_dataset(args.dataset, args.data_path, 'train')
        images_test, labels_test, label_names = read_image_dataset(args.dataset, args.data_path, 'test')
        run_core(args, images_train, labels_train, images_test, labels_test, label_names)

if __name__ == '__main__':
    main()
