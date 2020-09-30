'''
Created on Mar 28, 2017

@author: shujon
'''
import os
import pickle
import numpy as np, tensorflow as tf, tqdm
from tensorflow.examples.tutorials.mnist import input_data
import prettytensor as pt
#import matplotlib.pyplot as plt
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

@pt.Register
class custom_subtraction(pt.VarStoreMethod):
    def __call__(self, input_layer, matrix4,  name="cus_subtract"):
        with tf.variable_scope(name):
            
            conv = tf.subtract(input_layer, matrix4)
            conv1 = pt.wrap(tf.add(input_layer, matrix4))
            return conv
@pt.Register
class custom_add(pt.VarStoreMethod):
    def __call__(self, input_layer, matrix5, matrix,  name="cus_add"):
        with tf.variable_scope(name):
            
            conv = tf.add(input_layer, matrix)
            conv1 = pt.wrap(tf.subtract(input_layer, matrix))
            return conv, conv1

def sub_temp():
    matrix = tf.constant([[2.]])
    template= \
        (pt.template("input").  # 128*9*4*4
         custom_add(pt.UnboundVariable("hidden"), matrix)
         )        
    return template

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
matrix3 = tf.constant([[2.]])
product = pt.wrap(tf.matmul(matrix1, matrix2))
matrix4 = pt.wrap(tf.constant([[2.]]))
node1 = (product.apply(tf.add, matrix3).custom_subtraction(matrix4).apply(tf.add, matrix3))

matrix5 = pt.wrap(tf.constant([[3.]]))
sub_template = sub_temp()

node2, matrix5= sub_template.construct(input=node1, hidden=matrix5)

node2, matrix5 = sub_template.construct(input=node2, hidden=matrix5)
sess = tf.Session()

result = sess.run(matrix5)
print(result)
sess.close()