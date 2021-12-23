# import os
# from pathlib import Path
# from tensorflow.keras.callbacks import ModelCheckpoint
# from model_test import MusicTransformer
# from custom.layers import *
# from custom import callback
# import params as par
# from tensorflow.keras.optimizers import Adam
from data import DataNew
import tensorflow as tf
# import utils
# import argparse
# import datetime
# import midi_processor.processor as sequence
import numpy as np

dataset = DataNew('midi_processed', 8, 4)

# g = [el for el in dataset.generators_dict["train"]]
#
a = np.array([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]])
b = np.array([[20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43], [50, 51, 52, 53]])
# # b -
def concat_parallel2(a, b):
    # a = np.array([a], dtype=np.int32)
    # a = tf.convert_to_tensor(a, dtype=tf.int32)
    # b = np.array([b], dtype=np.int32)
    # b = tf.convert_to_tensor(b, dtype=tf.int32)
    l = [(tf.convert_to_tensor(np.array([x], dtype=np.int32)),
         tf.convert_to_tensor(np.array([y], dtype=np.int32))) for x in a for y in b]
    return l

def concat_parallel(a, b):
    a = list(a)
    b = list(b)
    z = list(zip(a, b))
    l = [(tf.convert_to_tensor(np.array([x], dtype=np.int32)),
         tf.convert_to_tensor(np.array([y], dtype=np.int32))) for x, y in z]
    return l
#
x = concat_parallel(a, b)
print(x)
# print(x[0])
#
#
    

# x = np.stack(a, b)
# print(x)
for el in dataset.generators_dict["test"]:
    print(el[0])
    print(el[1])
    # print(el)
    print(len(el))


#     print(len(el[0]))
#     # print(str(el))
# print(i)
# def reshape_trimmed(l, x, y):
#     print(len(l))
#     to_trim = len(l) % y
#     l = l[0:-to_trim]
#     return l.reshape(x, y)
#
#
#
# l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# data = np.array(l)
# data_sp = reshape_trimmed(data, -1, 3)
#
# # data_sp = data.reshape(-1, 3)
# print(data_sp)

# print(len(dataset[0][0][0]))