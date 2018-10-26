from utils import *
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np

class LeNet():
    def __init__(self,x, mu, sigma, bias_value,
                 conv1_params, conv2_params, p1_params,
                 p2_params, fc1_params, fc2_params, fc3_params, hold_prob):
        self.mu = mu
        self.sigma = sigma
        self.x = x
        self.conv1_params = conv1_params
        self.conv2_params = conv2_params
        self.p1_params = p1_params
        self.p2_params = p2_params
        self.fc1_params = fc1_params
        self.fc2_params = fc2_params
        self.fc3_params = fc3_params
        self.bias_value = bias_value
        self.hold_prob = hold_prob
        self.logits = self._build_neural_net()
        
        
    
    def _build_neural_net(self):
        
        filter_height_c1, filter_width_c1, channel_in_c1, channel_out_c1, \
        stride_c1, padding_c1 = self.conv1_params
        
        shape_1 = [filter_height_c1, filter_width_c1, channel_in_c1, channel_out_c1]
        
        # First layer: convolutional layer
        conv_1 = convolutional_layer(self.x, shape_1,
                                     self.mu, self.sigma, self.bias_value,
                                     stride_c1, padding_c1, name="conv_1")
        
        # First layer: pooling layer
        ksize_p1, stride_p1, padding_p1 = self.p1_params
        conv_1_pooling = max_pooling(conv_1, ksize_p1, stride_p1, padding_p1, name="max_pool_1")
        
        # Second layer: convolutional layer
        filter_height_c2, filter_width_c2, channel_in_c2, channel_out_c2, \
        stride_c2, padding_c2 = self.conv2_params
        shape_2 = [filter_height_c2, filter_width_c2, channel_in_c2, channel_out_c2]

        conv_2 = convolutional_layer(conv_1_pooling, shape_2,
                                     self.mu, self.sigma, self.bias_value,
                                     stride_c2, padding_c2, name="conv_2")
        
        # Second layer: pooling layer
        ksize_p2, stride_p2, padding_p2 = self.p2_params
        conv_2_pooling = max_pooling(conv_2, ksize_p2, stride_p2, padding_p2, name="max_pool_2")
        
        # FLatten layer
        #width, height, channel = conv_2_pooling.get_shape()[1:]
        shape_4 =conv_2_pooling.get_shape().as_list()  # a list: [None, 9, 2]
        dim = np.prod(shape_4[1:])
        #dimension = width*height*channel
        conv_flat = tf.reshape(conv_2_pooling, [-1, dim])
        
        # fully connected layer 1
        output_activations_1 = self.fc1_params
        fully_connected_layer_1 = tf.nn.relu(fully_connected_layer(conv_flat, output_activations_1, self.mu, self.sigma, self.bias_value,
                                                                   name="fully_connected_1"))
        # dropout
        fully_connected_layer_1 = tf.nn.dropout(fully_connected_layer_1, keep_prob=self.hold_prob)
        
        # fully connected layer 2
        output_activations_2 = self.fc2_params
        fully_connected_layer_2 = tf.nn.relu(
            fully_connected_layer(fully_connected_layer_1, output_activations_2, self.mu, self.sigma, self.bias_value,
                                  name="fully_connected_2" ))
        fully_connected_layer_2 = tf.nn.dropout(fully_connected_layer_2, keep_prob=self.hold_prob)
        
        # fully connected layer 3
        output_activations_3 = self.fc3_params
        fully_connected_layer_3 = fully_connected_layer(fully_connected_layer_2, output_activations_3, self.mu, self.sigma, self.bias_value,
                                                        name="fully_connected_3")
        
        return fully_connected_layer_3
    
    def get_logits(self):
        return self.logits
    
    def get_summary(self):
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        
    
        
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    