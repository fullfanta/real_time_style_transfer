import tensorflow as tf
from network_common import *

def residual(input, name, input_channel, reuse):
    with tf.variable_scope(name, reuse) as scope:
        with tf.variable_scope('res_conv1', reuse) as scope:
            W_conv = weight_variable([ 3, 3, input_channel, input_channel ])
            conv = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0,0]], mode='REFLECT')
            conv = conv2d(conv, W_conv, stride = 1, padding='VALID')
            conv = inst_norm(conv)
            conv = tf.nn.relu(conv)

        with tf.variable_scope('res_conv2', reuse) as scope:
            W_conv = weight_variable([ 3, 3, input_channel, input_channel ])
            conv = tf.pad(conv, [[0, 0], [1, 1], [1, 1], [0,0]], mode='REFLECT')
            conv = conv2d(conv, W_conv, stride = 1, padding='VALID')
            conv = inst_norm(conv)

        conv = conv + input

    return conv
        
        

class TransformNetwork:
    def __init__(self, name='name'):
        self.name=name

    def inference(self, image, reuse = False, train = True):
        with tf.variable_scope(self.name, reuse) as scope:
            if reuse:
                scope.reuse_variables()

            batch_size = tf.shape(image)[0]

            with tf.variable_scope('conv1', reuse) as scope:
                W_conv = weight_variable([ 9, 9, 3, 32 ])
                conv = tf.pad(image, [[0, 0], [4, 4], [4, 4], [0,0]], mode='REFLECT')
                conv = conv2d(conv, W_conv, stride = 1, padding='VALID')
                conv = inst_norm(conv)
                conv1 = tf.nn.relu(conv)

                print 'conv1 ', conv1.get_shape()

            with tf.variable_scope('conv2', reuse) as scope:
                W_conv = weight_variable([ 3, 3, 32, 64 ])
                conv = tf.pad(conv1, [[0, 0], [1, 1], [1, 1], [0,0]], mode='REFLECT')
                conv = conv2d(conv, W_conv, padding='VALID')
                conv = inst_norm(conv)
                conv2 = tf.nn.relu(conv)

                print 'conv2 ', conv2.get_shape()

            with tf.variable_scope('conv3', reuse) as scope:
                W_conv = weight_variable([ 3, 3, 64, 128 ])
                conv = tf.pad(conv2, [[0, 0], [1, 1], [1, 1], [0,0]], mode='REFLECT')
                conv = conv2d(conv, W_conv, padding='VALID')
                conv = inst_norm(conv)
                conv3 = tf.nn.relu(conv)

                print 'conv3 ', conv3.get_shape()

            residual1 = residual(conv3, 'residual1', 128, reuse)
            print 'residual1 -', residual1.get_shape()

            residual2 = residual(residual1, 'residual2', 128, reuse)
            print 'residual2 -', residual2.get_shape()

            residual3 = residual(residual2, 'residual3', 128, reuse)
            print 'residual3 -', residual3.get_shape()


            ######
            # deconvolution

            with tf.variable_scope('deconv3', reuse) as scope:
                W_conv = weight_variable([ 3, 3, 128, 64 ])
                shape = tf.shape(conv2)
                conv = tf.image.resize_images(residual3, [shape[1], shape[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                conv = tf.pad(conv, [[0, 0], [1, 1], [1, 1], [0,0]], mode='REFLECT')
                conv = conv2d(conv, W_conv, stride=1, padding='VALID')
                #conv = deconv2d(residual3, W_conv, shape)
                #conv = tf.reshape(conv, shape)
                conv = inst_norm(conv)
                conv = tf.nn.relu(conv)

                print 'deconv3 - ', conv.get_shape()

            with tf.variable_scope('deconv2', reuse) as scope:
                W_conv = weight_variable([ 3, 3, 64, 32 ])
                shape = tf.shape(conv1)
                conv = tf.image.resize_images(conv, [shape[1], shape[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                conv = tf.pad(conv, [[0, 0], [1, 1], [1, 1], [0,0]], mode='REFLECT')
                conv = conv2d(conv, W_conv, stride=1, padding='VALID')
                #conv = deconv2d(conv, W_conv, shape)
                #conv = tf.reshape(conv, shape)
                conv = inst_norm(conv)
                conv = tf.nn.relu(conv)

                print 'deconv2 - ', conv.get_shape()

            with tf.variable_scope('deconv1', reuse) as scope:
                W_conv = weight_variable([ 9, 9, 32, 3 ])
                conv = tf.image.resize_images(conv, [shape[1], shape[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                conv = tf.pad(conv, [[0, 0], [4, 4], [4, 4], [0,0]], mode='REFLECT')
                conv = conv2d(conv, W_conv, stride=1, padding='VALID')
                #conv = tf.nn.tanh(conv) * 150.0 + 255.0 / 2
                #conv = (tf.nn.tanh(conv) * 255.0 + 255.0) / 2
                conv = tf.nn.tanh(conv) * 255.0 + 255.0
                conv = tf.div(conv, 2, name='output')

                print 'deconv1 - ', conv.get_shape()

            return conv
