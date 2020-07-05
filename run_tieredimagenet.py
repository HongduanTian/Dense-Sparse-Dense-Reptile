"""
Train a model on miniImageNet.
"""

import random
import os
import tensorflow as tf

import pandas as pd
from DSD.args import argument_parser, model_kwargs, baseline_train_kwargs, DSD_train_kwargs, IHT_train_kwargs, evaluate_kwargs, data_kwargs
from DSD.eval import evaluate
from DSD.models import MiniImageNetModel
from DSD.tieredimagenet import dataset_tiered
from DSD.basetrain import base_train
from DSD.dsdtrain import dsd_train
from DSD.ihttrain import iht_train


#DATA_DIR = 'data/tiered-imagenet'

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    random.seed(args.seed)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    train_set = dataset_tiered(split='train', **data_kwargs(args))
    train_set.load_data_pkl()

    test_set = dataset_tiered(split='test', **data_kwargs(args))
    test_set.load_data_pkl()

    model = MiniImageNetModel(args.classes, args.num_filters, **model_kwargs(args))

    # set result file name
    if args.sparse_mode == 'Base':
        exp_string = 'TieredImageNet' + \
                     str(args.classes) + '-way' + \
                     str(args.shots) + '-shot' + \
                     str(args.num_filters) + 'channels' + \
                     str(args.sparse_mode) + 'mode'
    elif args.sparse_mode == 'DSD':
        exp_string = 'TieredImageNet' + \
                     str(args.classes) + '-way' + \
                     str(args.shots) + '-shot' + \
                     str(args.num_filters) + 'channels' + \
                     str(args.sparse_mode) + 'mode' + \
                     '_compress_rate' + str(args.compress_rate) + \
                     '_sparse_iter' + str(args.sparse_iter) + \
                     '_retrain_iter' + str(args.retrain_iter)
    elif args.sparse_mode == 'IHT':
        exp_string = 'TieredImageNet' + \
                     str(args.classes) + '-way' + \
                     str(args.shots) + '-shot' + \
                     str(args.num_filters) + 'channels' + \
                     str(args.sparse_mode) + 'mode' + \
                     '_compress_rate' + str(args.compress_rate) + \
                     '_interval' + str(args.sparse_interval) + \
                     '_ratio' +str(args.pr_ratio)
    else:
        raise ValueError("No such sparse mode, please check your sparse mode.")

    with tf.Session(config=config) as sess:
        if not args.pretrained:
            print('Training...')

            if args.sparse_mode == 'DSD':
                print("###################** DSD Mode **###################")
                dsd_train(sess, model, train_set, test_set, args.checkpoint, **DSD_train_kwargs(args))
            elif args.sparse_mode == 'IHT':
                print("###################** IHT Mode **###################")
                iht_train(sess, model, train_set, test_set, args.checkpoint, **IHT_train_kwargs(args))
            elif args.sparse_mode == 'Base':
                print("###################** Base Mode **###################")
                base_train(sess, model, train_set, test_set, args.checkpoint, **baseline_train_kwargs(args))
            else:
                raise  ValueError("Unrecognized sparse mode.")
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)

        train_accuracy, train_ci95= evaluate(sess, model, train_set, **eval_kwargs)
        print('Train accuracy: {:.4f}(pm){:.4f}'.format(train_accuracy, train_ci95))
        test_accuracy, test_ci95= evaluate(sess, model, test_set, **eval_kwargs)
        print('Test accuracy: {:.4f}(pm){:.4f}'.format(test_accuracy, test_ci95))

        dataframe = pd.DataFrame({'training accuracy': [train_accuracy], 'testing accuracy': [test_accuracy],
                                  'train ci95':[train_ci95], 'test ci95':[test_ci95]})
        dataframe.to_csv(exp_string + '.csv', index=False, sep=',')

if __name__ == '__main__':
    main()