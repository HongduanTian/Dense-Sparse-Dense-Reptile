import numpy as np
import tensorflow as tf

class Mask:

    def __init__(self, session, num_filters, num_classes):

        self._session = session
        #self.compress_rate = compress_rate   # For IHT case, here is the final compress rate to reach
        self._num_classes = num_classes
        self.layer_to_prune_index = [4, 8, 12, 16]
        self._num_filters = num_filters

    def mask_generator(self, var_list, cur_rate):
        '''
        Used to generate mask matrix of the whole parameters matrix
        :param params_list:
        :return: a mask for entire network
        '''

        params_list = var_list
        mask_temp = []
        for index in range(len(self.layer_to_prune_index)):

            threshold = self.compute_threshold(params_list[self.layer_to_prune_index[index]], cur_rate)
            if (index < (len(self.layer_to_prune_index)-1)):
                I_matrix = np.ones(shape=(3, 3, self._num_filters, self._num_filters))
            else:
                I_matrix = np.ones(shape=(len(params_list[self.layer_to_prune_index[index]]), self._num_classes))
            mask_temp.append((np.abs(params_list[self.layer_to_prune_index[index]])>threshold) * I_matrix)

        return mask_temp

    def compute_threshold(self, var_list, rate):
        '''
        Used to compute the mask matrix of each layer, as referred in Han et al.(2016)
        :param var_list, rate(pruning rate)
        :return: a threshold used to generate the binary mask.
        '''
        var_tensor = tf.convert_to_tensor(var_list)
        length_var = int(np.prod(var_tensor.get_shape()))

        # sort operation and compute threshold
        abs_weights = np.abs(var_list)
        abs_weights_tensor = tf.convert_to_tensor(abs_weights)
        weights_vector = tf.reshape(abs_weights_tensor,
                                    shape=(int(np.prod(abs_weights_tensor.get_shape())), 1))
        abs_weights_sort = np.sort(self._session.run(weights_vector), axis=0)
        threshold = abs_weights_sort[int(length_var * rate)]

        return threshold