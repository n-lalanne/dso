#!/usr/bin/env python3
import os
import json
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(script_dir, "error_with_min_max.json")) as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    mapping_keys = list(map(float, data['mapping'].keys()))
    mapping_values = list(map(float, data['mapping'].values()))

    mapping = {
        'x1': float(mapping_keys[0]),
        'x2': float(mapping_keys[1]),
        'y1': float(mapping_values[0]),
        'y2': float(mapping_values[1]),
    }
    
    mapping_gradient = (mapping['y2'] - mapping['y1']) / (mapping['x2'] - mapping['x1'])
    def map_value(value):
        if type(value) in (float, int):
            return mapping['y1'] + mapping_gradient * (value - mapping['x1'])
        elif type(value) in (tuple, list):
            return tuple(map(
                lambda value: mapping['y1'] + mapping_gradient * (value - mapping['x1']),
                value
            ))

    def map_intervals(intervals):
        return tuple((i * mapping_gradient for i in intervals))

    errors = data["errors"]
    
    sequences = list(errors.keys())
    dso_error = list(map(map_value, map(lambda x: x[0][1], errors.values())))
    dsvio_error = list(map(map_value, map(lambda x: x[1][1], errors.values())))
    
    dso_error_bar = np.transpose(np.asmatrix(list(
        map(map_intervals, map(lambda x: (x[0][1] - x[0][0], x[0][2] - x[0][1]), errors.values()))
    )))
    dsvio_error_bar = np.transpose(np.asmatrix(list(
        map(map_intervals, map(lambda x: (x[1][1] - x[1][0], x[1][2] - x[1][1]), errors.values()))
    )))
    
    print('dso_error', dso_error)
    print('dsvio_error', dsvio_error)
    
    print('dso_error_bar', dso_error_bar)
    print('dsvio_error_bar', dsvio_error_bar)
    
    w = 0.3
    capsize = 5

    x = np.asarray(range(1, len(sequences)+1))

    plt.rcParams.update({'font.size': 28, 'font.family': 'Times New Roman'})

    plt.bar(
        x - w/2, 
        dso_error,
        yerr=dso_error_bar.tolist(),
        label='DSO',
        color='r',
        align='center',
        width=w,
        capsize=capsize
    )
    
    plt.bar(
        x + w/2, 
        dsvio_error,
        yerr=dsvio_error_bar.tolist(),
        label='DSVIO',
        color='b',
        align='center',
        width=w,
        capsize=capsize
    )
    plt.xticks(x, sequences)
    plt.yticks(np.arange(0, 0.55, 0.05))
    plt.legend()
    
    plt.show()

