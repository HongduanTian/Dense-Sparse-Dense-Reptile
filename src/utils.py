import numpy as np
import tensorflow as tf

def update_params(params, mask):

    params_to_process = params
    layer_to_process_index = [4, 8, 12, 16]

    for index in range(len(layer_to_process_index)):
        params_to_process[layer_to_process_index[index]] = params_to_process[layer_to_process_index[index]] * mask[index]

    return params_to_process