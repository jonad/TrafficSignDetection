import tensorflow as tf
import numpy as np

def init_weights(shape, mu, sigma):
    '''
    Initialize weight parameters with a random distribution of dimension 'shape',
    mean 'mu', and standard deviation 'sigma'
    :param shape: shape
    :param mu: mean
    :param sigma: standard deviation
    :return:
    '''
    init_weights_vals = tf.truncated_normal(shape=shape, mean=mu, stddev=sigma)
    return tf.Variable(init_weights_vals)

def init_bias(shape, value):
    '''
    Initialize the bias parameters of shape 'shape' with value 'value'
    :param shape: shape
    :param value: value
    :return:
    '''
    init_bias_vals = tf.constant(value, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(input, filters, stride, padding):
    '''
    Computes a 2-D convolution given 4-D input and a filter tensors
    :param input:
    :param filters:
    :param stride:
    :param padding:
    :return:
    '''
    return tf.nn.conv2d(input, filters, strides=[1, stride, stride, 1], padding=padding)

def max_pooling(input, ksize, stride, padding, name="max_pooling"):
    '''
    
    :param input:
    :param size:
    :param stride:
    :param padding:
    :return:
    '''
    with tf.name_scope(name):
        return tf.nn.max_pool(input, ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding)

def convolutional_layer(input, shape, mu, sigma, value, stride, padding, name="conv"):
    '''
    
    :param input:
    :param shape:
    :param mu:
    :param sigma:
    :param value:
    :param stride:
    :param padding:
    :return:
    '''
    with tf.name_scope(name):
        W = init_weights(shape, mu, sigma)
        b = init_bias([shape[3]], value)
        conv = conv2d(input, W, stride, padding)
        activation = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", activation)
        return activation


def fully_connected_layer(input, size, mu, sigma, value, name="fully_connected"):
    '''
    
    :param input:
    :param size:
    :param mu:
    :param sigma:
    :param value:
    :return:
    '''
    with tf.name_scope(name):
        input_size = int(input.get_shape()[1])
        W = init_weights([input_size, size], mu, sigma)
        b = init_bias([size], value)
        activation = tf.matmul(input, W) + b
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", activation)
        return activation
    
def evaluate(X_data, y_data, evaluation_operation, batch_size):
    num_examples = len(X_data)
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
    y = tf.placeholder(tf.int32, shape=(None))
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        accuracy = sess.run(evaluation_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def normalize_and_grayscale(x):
    
    return np.sum((x / 255.) * [0.299, 0.587, 0.114], axis=-1, keepdims=True)
    
    
    


    
    
    
    
    
    
    