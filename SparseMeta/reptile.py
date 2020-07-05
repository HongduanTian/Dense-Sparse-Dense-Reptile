"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""

import random

import tensorflow as tf

from .variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)

class Reptile:
    """
    A meta-learning session.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, session, variables=None, transductive=False, pre_step_op=None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._transductive = transductive
        self._pre_step_op = pre_step_op

        self._sparse = False
        self.mask = None
        self.layer_index = [4, 8, 12, 16]

    # pylint: disable=R0913,R0914
    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size,
                   dataset_name):
        """
        Perform a Reptile training step.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
        """
        old_vars = self._model_state.export_variables()

        new_vars = []
        for _ in range(meta_batch_size):
            if dataset_name == 'miniimagenet' or dataset_name == 'omniglot':
                mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            elif dataset_name == 'tieredimagenet':
                mini_dataset = dataset.next_data(num_classes, num_shots)
            else:
                raise ValueError(" Unrecognized dataset. ")
            for batch in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement, dataset_name):
                if self._sparse:
                    current_params = self._model_state.export_variables()
                    processed_params = self.update_params(current_params, self.mask)
                    self._model_state.import_variables(processed_params)

                inputs, labels = zip(*batch)
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)
        new_vars = average_vars(new_vars)
        new_kernel_params = interpolate_vars(old_vars, new_vars, meta_step_size)

        if self._sparse:
            return_params = self.update_params(new_kernel_params, self.mask)
        else:
            return_params = new_kernel_params
        self._model_state.import_variables(return_params)

    def evaluate(self,
                 dataset,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement,
                 dataset_name):
        """
        Run a single evaluation of the model.

        Samples a few-shot learning task and measures
        performance.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          predictions: a Tensor of integer label predictions.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.

        Returns:
          The number of correctly predicted samples.
            This always ranges from 0 to num_classes.
        """
        if dataset_name == 'miniimagenet' or dataset_name == 'omniglot':
            mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
        elif dataset_name == 'tieredimagenet':
            mini_dataset = dataset.next_data(num_classes, num_shots+1)
        else:
            raise ValueError(" Unrecognized dataset. ")
        train_set, test_set = _split_train_test(mini_dataset, dataset_name)

        old_vars = self._full_state.export_variables()
        for batch in _mini_batches(train_set, inner_batch_size, inner_iters, replacement, dataset_name):
            if self._sparse:
                current_params = self._model_state.export_variables()
                processed_params = self.update_params(current_params, self.mask)
                self._model_state.import_variables(processed_params)
            inputs, labels = zip(*batch)
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
        test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
        num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        self._full_state.import_variables(old_vars)
        return num_correct

    def _test_predictions(self, train_set, test_set, input_ph, predictions):
        if self._transductive:
            inputs, _ = zip(*test_set)
            return self.session.run(predictions, feed_dict={input_ph: inputs})
        res = []
        for test_sample in test_set:
            inputs, _ = zip(*train_set)
            inputs += (test_sample[0],)
            res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
        return res

    def get_mask(self, mask):
        self.mask = mask

    def update_sparse_signal(self, signal):
        self._sparse = signal

    def update_params(self, var_list, mask):
        index = self.layer_index
        var = var_list
        for i in range(len(index)):
            var[index[i]] = var[index[i]] * mask[i]
        return var

def _sample_mini_dataset(dataset, num_classes, num_shots):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        for sample in class_obj.sample(num_shots):
            yield (sample, class_idx)

def _mini_batches(samples, batch_size, num_batches, replacement, dataset_name):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    if dataset_name == 'miniimagenet':
        samples = list(samples)
        if replacement:
            for _ in range(num_batches):
                yield random.sample(samples, batch_size)
            return

    cur_batch = []
    batch_count = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return

def _split_train_test(train_set, dataset_name='tieredimagenet', test_shots=1):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    if dataset_name == 'miniimagenet':
        train_set = list(train_set)

    test_set = []
    labels = set(item[1] for item in train_set)
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(train_set):
                if item[1] == label:
                    del train_set[i]
                    test_set.append(item)
                    break
    if len(test_set) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')
    return train_set, test_set
