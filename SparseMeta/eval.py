"""
Helpers for evaluating models.
"""
import numpy as np
from tqdm import tqdm
from .reptile import Reptile
from .variables import weight_decay

# pylint: disable=R0913,R0914
def evaluate(sess,
             model,
             dataset,
             num_classes=5,
             num_shots=5,
             eval_inner_batch_size=5,
             eval_inner_iters=50,
             replacement=False,
             num_samples=10000,
             transductive=False,
             weight_decay_rate=1,
             reptile_fn=Reptile,
             dataset_name='tieredimagenet'):
    """
    Evaluate a model on a dataset.
    """
    reptile = reptile_fn(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    # total_correct = 0
    all_accuracy = []
    for _ in tqdm(range(num_samples), desc="Evaluating Phase"):
        correct = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                   model.minimize_op, model.predictions,
                                   num_classes=num_classes, num_shots=num_shots,
                                   inner_batch_size=eval_inner_batch_size,
                                   inner_iters=eval_inner_iters, replacement=replacement, dataset_name=dataset_name)
        cur_acc = correct / num_classes
        all_accuracy.append(cur_acc)

    avg_acc = np.mean(all_accuracy)
    std = np.std(all_accuracy)
    ci95 = 1.96 * std / np.sqrt(num_samples)
    return avg_acc, ci95
