"""
Command-line argument parsing.
"""

import argparse
from .reptile import Reptile
from functools import partial

import tensorflow as tf
def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrained', help='evaluate a pre-trained model',
                        action='store_true', default=False)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--checkpoint', help='checkpoint directory', default='model_checkpoint')
    parser.add_argument('--classes', help='number of classes per inner task', default=5, type=int)
    parser.add_argument('--shots', help='number of examples per class', default=5, type=int)
    parser.add_argument('--train-shots', help='shots in a training batch', default=0, type=int)
    parser.add_argument('--inner-batch', help='inner batch size', default=5, type=int)
    parser.add_argument('--inner-iters', help='inner iterations', default=20, type=int)
    parser.add_argument('--replacement', help='sample with replacement', action='store_true')
    parser.add_argument('--learning-rate', help='Adam step size', default=1e-3, type=float)
    parser.add_argument('--meta-step', help='meta-training step size', default=0.1, type=float)
    parser.add_argument('--meta-step-final', help='meta-training step size by the end',
                        default=0.1, type=float)
    parser.add_argument('--meta-batch', help='meta-training batch size', default=1, type=int)
    parser.add_argument('--meta-iters', help='meta-training iterations', default=400000, type=int)
    parser.add_argument('--eval-batch', help='eval inner batch size', default=5, type=int)
    parser.add_argument('--eval-iters', help='eval inner iterations', default=50, type=int)
    parser.add_argument('--eval-samples', help='evaluation samples', default=10000, type=int)
    parser.add_argument('--eval-interval', help='train steps per eval', default=10, type=int)
    parser.add_argument('--weight-decay', help='weight decay rate', default=1, type=float)
    parser.add_argument('--transductive', help='evaluate all samples at once', action='store_true')
    parser.add_argument('--foml', help='use FOML instead of Reptile', action='store_true')
    parser.add_argument('--foml-tail', help='number of shots for the final mini-batch in FOML',
                        default=None, type=int)
    parser.add_argument('--sgd', help='use vanilla SGD instead of Adam', action='store_true')
    parser.add_argument('--DATASET', type=str, default="miniimagenet", help='datasetname, can be omniglot, miniimagenet or tieredimagenet.)
    parser.add_argument('--img-size', type=int, default=84, 
                        help='used to adjust the image size when running miniimagenet and tieredimagenet dataset')
    '---------------------------------------------compress rate-----------------------------------------------------'
    parser.add_argument('--gpu-id', type=str, default='1', help='set the GPU ID')
    parser.add_argument('--num_filters', type=int, default=32, help='choose the filters number used in network.')
    parser.add_argument('--sparse-mode', type=str, default='Base', help='here can be Base or IHT or DSD')
    parser.add_argument('--compress-rate', type=float, default=0.5, help='pruning rate of the network')
    parser.add_argument('--sparse-iter', type=int, default=2, help='iteration number when to prune the network.')
    # DSD mode args
    parser.add_argument('--retrain-iter', type=int, default=90000, help='iteration number when to retrain the network.')
    # IHT mode args
    parser.add_argument('--sparse-interval', type=int, default=20000, help='interval of re-generate new mask')
    parser.add_argument('--init-rate', type=float, default=0.1, help='set the initial pruning rate')
    parser.add_argument('--ratio', type=float, default=0.75, helpd='the ratio of pruning rate in an IHT interval')
    '--------------------------------------------------tiredimagenet args-------------------------------------------------------'
    parser.add_argument('--x_dim', type=str, default="84,84,3", metavar='XDIM',help='input image dims')
    parser.add_argument('--pkl', type=int, default=1, metavar='PKL', help="1 for use pkl dataset, 0 for original images")
    parser.add_argument('--DATA_DIR', type=str, default="/home/tianhongduan/icml/SparseMeta/data/tieredimagenet", help='dataset path')
    parser.add_argument('--tiered-ratio', type=float, default=1.0, metavar='RATIO', help="ratio of labeled data (for semi-supervised setting")
    return parser

def model_kwargs(parsed_args):
    """
    Build the kwargs for model constructors from the
    parsed command-line arguments.
    """
    res = {'learning_rate': parsed_args.learning_rate}
    if parsed_args.sgd:
        res['optimizer'] = tf.train.GradientDescentOptimizer
    return res

def data_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'x_dim':parsed_args.x_dim,
        'ratio': parsed_args.tiered_ratio,
        'seed':parsed_args.seed,
        'DATA_DIR':parsed_args.DATA_DIR
    }

def baseline_train_kwargs(parsed_args):
    """
    Build kwargs for the baseline mode train() function from the parsed
    command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'train_shots': (parsed_args.train_shots or None),
        'inner_batch_size': parsed_args.inner_batch,
        'inner_iters': parsed_args.inner_iters,
        'replacement': parsed_args.replacement,
        'meta_step_size': parsed_args.meta_step,
        'meta_step_size_final': parsed_args.meta_step_final,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'weight_decay_rate': parsed_args.weight_decay,
        'transductive': parsed_args.transductive,
        'reptile_fn': _args_reptile(parsed_args),
        'dataset_name': parsed_args.DATASET
    }

def DSD_train_kwargs(parsed_args):
    """
        Build kwargs for the DSD mode train() function from the parsed
        command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'train_shots': (parsed_args.train_shots or None),
        'inner_batch_size': parsed_args.inner_batch,
        'inner_iters': parsed_args.inner_iters,
        'replacement': parsed_args.replacement,
        'meta_step_size': parsed_args.meta_step,
        'meta_step_size_final': parsed_args.meta_step_final,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'weight_decay_rate': parsed_args.weight_decay,
        'transductive': parsed_args.transductive,
        'reptile_fn': _args_reptile(parsed_args),
        'num_filters': parsed_args.num_filters,
        'compress_rate': parsed_args.compress_rate,
        'sparse_iter': parsed_args.sparse_iter,
        'retrain_iter': parsed_args.retrain_iter,
        'dataset_name': parsed_args.DATASET
    }

def IHT_train_kwargs(parsed_args):
    """
        Build kwargs for the baseline mode train() function from the parsed
        command-line arguments.
        """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'train_shots': (parsed_args.train_shots or None),
        'inner_batch_size': parsed_args.inner_batch,
        'inner_iters': parsed_args.inner_iters,
        'replacement': parsed_args.replacement,
        'meta_step_size': parsed_args.meta_step,
        'meta_step_size_final': parsed_args.meta_step_final,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'weight_decay_rate': parsed_args.weight_decay,
        'transductive': parsed_args.transductive,
        'reptile_fn': _args_reptile(parsed_args),
        'num_filters': parsed_args.num_filters,
        'compress_rate': parsed_args.compress_rate,
        'sparse_iter': parsed_args.sparse_iter,
        'sparse_interval': parsed_args.sparse_interval,
        'init_rate': parsed_args.init_rate,
        'dataset_name': parsed_args.DATASET,
        'ratio': parsed_args.ratio
    }

# only one mode for evaluation
def evaluate_kwargs(parsed_args):
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'replacement': parsed_args.replacement,
        'weight_decay_rate': parsed_args.weight_decay,
        'num_samples': parsed_args.eval_samples,
        'transductive': parsed_args.transductive,
        'reptile_fn': _args_reptile(parsed_args),
        'dataset_name': parsed_args.DATASET
    }

def _args_reptile(parsed_args):

    return Reptile
