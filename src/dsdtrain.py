import os
import time

import tensorflow as tf

from tqdm import tqdm

from .reptile import Reptile
from .variables import weight_decay
from .mask_generator import Mask
from .variables import VariableState
from .utils import update_params

# pylint: disable=R0913,R0914
def dsd_train(sess,
          model,
          train_set,
          test_set,
          save_dir,
          num_filters=64,
          compress_rate=0.5,
          sparse_iter=500,
          retrain_iter=800,
          num_classes=5,
          num_shots=5,
          inner_batch_size=5,
          inner_iters=20,
          replacement=False,
          meta_step_size=0.1,
          meta_step_size_final=0.1,
          meta_batch_size=1,
          meta_iters=400000,
          eval_inner_batch_size=5,
          eval_inner_iters=50,
          eval_interval=10,
          weight_decay_rate=1,
          time_deadline=None,
          train_shots=None,
          transductive=False,
          reptile_fn=Reptile,
          dataset_name='miniimagenet',
          log_fn=print):
    """
    Train a model on a dataset.
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
    reptile = reptile_fn(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))

    # initialize the mask object
    mask_class = Mask(session=sess,
                      num_filters=num_filters,
                      num_classes=num_classes)
    # initialize a model_state to process the kernel
    model_state = VariableState(session=sess, variables=tf.trainable_variables())

    accuracy_ph = tf.placeholder(tf.float32, shape=())
    tf.summary.scalar('accuracy', accuracy_ph)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'test'), sess.graph)
    tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())
    for i in tqdm(range(meta_iters), desc="Training Phase"):

        if (i == sparse_iter):
            # generate mask & process params & set the signal in reptile
            current_params = model_state.export_variables()
            mask = mask_class.mask_generator(current_params, compress_rate)
            #print("The mask is {}".format(mask))
            reptile.get_mask(mask)
            reptile.update_sparse_signal(True)
            print("Mask Process Done...")
            processed_params = update_params(current_params, mask)
            model_state.import_variables(processed_params)

        if (i == retrain_iter):
            reptile.update_sparse_signal(False)
            print("Retraining Starting...")

        frac_done = i / meta_iters
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        reptile.train_step(train_set, model.input_ph, model.label_ph, model.minimize_op,
                           num_classes=num_classes, num_shots=(train_shots or num_shots),
                           inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                           replacement=replacement,
                           meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size, dataset_name=dataset_name)

        if i % eval_interval == 0:
            accuracies = []

            for dataset, writer in [(train_set, train_writer), (test_set, test_writer)]:
                correct = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                           model.minimize_op, model.predictions,
                                           num_classes=num_classes, num_shots=num_shots,
                                           inner_batch_size=eval_inner_batch_size,
                                           inner_iters=eval_inner_iters, replacement=replacement, dataset_name=dataset_name)
                summary = sess.run(merged, feed_dict={accuracy_ph: correct/num_classes})
                writer.add_summary(summary, i)
                writer.flush()
                accuracies.append(correct / num_classes)
            log_fn('batch %d: train=%f test=%f' % (i, accuracies[0], accuracies[1]))
        if i % 100 == 0 or i == meta_iters-1:
            saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=i)
        if time_deadline is not None and time.time() > time_deadline:
            break