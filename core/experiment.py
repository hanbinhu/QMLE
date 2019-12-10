import logging
logger = logging.getLogger("qmle")

def train(ctype, x, y, label_names, bond_inner, layer_channel, num_epoch):
    import numpy as np
    from .tree_tn import TTN
    model_list = []
    size = x.shape[1]
    channel = x.shape[3]
    bond_data = x.shape[4]
    if ctype == 'one-vs-all':
        n_class = len(label_names)
        n_each = y.shape[0]//n_class
        for i in range(n_class):
            model = TTN(size, channel, bond_data, bond_inner, 2, layer_channel)
            cx = x[n_each*i:n_each*(i+1),:,:,:,:]
            cy = y[n_each*i:n_each*(i+1),:]
            model.train(cx, cy, num_epoch)
            logger.info(f"{i+1}-th classifier's training for '{label_names[i]}' completed.'")

            p = model.predict(cx)
            cp = np.argmax(p,axis=1).tolist()
            n_correct = sum([cy[k][p] for k, p in enumerate(cp)])
            cacc = n_correct/n_each
            logger.info(f"{i+1}-th classifier's training accuracy for '{label_names[i]}': {cacc*100:.2f}%")
            model_list.append(model)
    elif ctype == 'one-hot':
        model = TTN(size, channel, bond_data, bond_inner, len(label_names), layer_channel)
        model.train(x, y, num_epoch)
        logger.info(f"One-hot classifier training completed.'")
        p = model.predict(x)
        pred = np.argmax(p,axis=1).tolist()
        n_correct = sum([y[i][p] for i, p in enumerate(pred)])
        acc = n_correct/y.shape[0]
        logger.info(f"One-hot classifier training accuracy: {acc*100:.2f}%")
        model_list.append(model)
    else:
        raise NotImplementedError
    return model_list

def test(model_list, x, y, label_names):
    import numpy as np
    n_tot = y.shape[0]
    if len(model_list) == 1:
        model = model_list[0]
        p = model.predict(x)
        pred = np.argmax(p,axis=1).tolist()
        n_correct = sum([y[i][p] for i, p in enumerate(pred)])
        acc = n_correct/n_tot
        logger.info(f"Overall testing accuracy: {acc*100:.2f}%")
    else:
        n_class = len(label_names)
        n_tot_correct = 0
        n_each = n_tot//n_class
        for i in range(n_class):
            model = model_list[i]
            cx = x[n_each*i:n_each*(i+1),:,:,:,:]
            cy = y[n_each*i:n_each*(i+1),:]
            p = model.predict(cx)
            cp = np.argmax(p,axis=1).tolist()
            n_correct = sum([cy[k][p] for k, p in enumerate(cp)])
            n_tot_correct += n_correct
            cacc = n_correct/n_each
            logger.info(f"{i+1}-th classifier's testing accuracy for '{label_names[i]}': {cacc*100:.2f}%")
        acc = n_correct/n_tot
        logger.info(f"Overall testing accuracy: {acc*100:.2f}%")
    return 0.0

def run_core(args, images_train, labels_train, images_test, labels_test, label_names):
    from .preprocess import preprocess_data
    x_train, y_train = preprocess_data(args.classifier_type,
            images_train, labels_train, len(label_names), args.num_train, args.bond_data)
    x_test, y_test = preprocess_data(args.classifier_type,
            images_test, labels_test, len(label_names), args.num_test, args.bond_data)

    logger.info('Data preparation completed.')

    model_list = train(args.classifier_type, x_train, y_train, label_names, args.bond_inner, args.layer_channel, args.num_epoch)

    logger.info('Training completed.')

    test(model_list, x_test, y_test, label_names)

    logger.info('Testing completed.')
