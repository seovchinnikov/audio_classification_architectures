from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
import keras as K
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, \
    Softmax, AveragePooling2D
from keras.engine.topology import get_source_inputs


def squeeze(axis):
    def f(x):
        import keras as K
        return K.backend.squeeze(x, axis=axis)
    return f


def expand_dims(axis):
    def f(x):
        import keras as K
        return K.backend.expand_dims(x, axis)

    return f

def JordiPons(vggish_params, num_filt, load_weights=True, weights='audioset',
              input_tensor=None, input_shape=None, include_top=False,
              out_dim=None, weights_path=None, pooling='avg'):
    '''
    An implementation of the JordiPons architecture.

    :param load_weights: if load weights
    :param weights: loads weights pre-trained on a preliminary version of YouTube-8M.
    :param input_tensor: input_layer
    :param input_shape: input data shape
    :param out_dim: output dimension


    :return: A Keras model instance.
    '''

    ker_init = 'glorot_uniform'#K.initializers.VarianceScaling()
    if weights not in {'audioset', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `audioset` '
                         '(pre-training on audioset).')

    if out_dim is None:
        out_dim = vggish_params.EMBEDDING_SIZE

    # input shape
    if input_shape is None:
        if vggish_params.EXAMPLE_WINDOW_SECONDS is not None:
            input_shape = (vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1)
        else:
            input_shape = (None, vggish_params.NUM_BANDS, 1)

    if input_tensor is None:
        aud_input = Input(shape=input_shape, name='input_1')
    else:
        if not K.backend.is_keras_tensor(input_tensor):
            aud_input = Input(tensor=input_tensor, shape=input_shape, name='input_1')
        else:
            aud_input = input_tensor

    # padding only time domain for an efficient 'same' implementation
    # (since we pool throughout all frequency afterwards)
    input_pad_7 = K.layers.ZeroPadding2D(padding=(3, 0))(aud_input)
    input_pad_3 = K.layers.ZeroPadding2D(padding=(1, 0))(aud_input)

    # [TIMBRE] filter shape 1: 7x0.9f
    conv1 = Conv2D(
        filters=num_filt,
        kernel_size=[7, int(0.9 * input_shape[1])],
        padding="valid",
        activation='relu',
        kernel_initializer=ker_init)(input_pad_7)

    bn_conv1 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv1)
    pool1 = K.layers.MaxPooling2D(pool_size=[1, int(bn_conv1.shape[2])],
                                  strides=[1, int(bn_conv1.shape[2])])(bn_conv1)
    # p1 = K.layers.Reshape((int(pool1.shape[1]), int(pool1.shape[3])))(pool1)
    p1 = K.layers.Lambda(squeeze(axis=2))(pool1)

    # [TIMBRE] filter shape 2: 3x0.9f
    conv2 = Conv2D(
        filters=num_filt * 2,
        kernel_size=[3, int(0.9 * input_shape[1])],
        padding="valid",
        activation='relu',
        kernel_initializer=ker_init)(input_pad_3)

    bn_conv2 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv2)
    pool2 = K.layers.MaxPooling2D(pool_size=[1, int(bn_conv2.shape[2])],
                                  strides=[1, int(bn_conv2.shape[2])])(bn_conv2)
    # p2 = K.layers.Reshape((int(pool2.shape[1]), int(pool2.shape[3])))(pool2)
    p2 = K.layers.Lambda(squeeze(axis=2))(pool2)

    # [TIMBRE] filter shape 3: 1x0.9f
    conv3 = Conv2D(
        filters=num_filt * 4,
        kernel_size=[1, int(0.9 * input_shape[1])],
        padding="valid",
        activation='relu',
        kernel_initializer=ker_init)(aud_input)

    bn_conv3 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv3)
    pool3 = K.layers.MaxPooling2D(pool_size=[1, int(bn_conv3.shape[2])],
                                  strides=[1, int(bn_conv3.shape[2])])(bn_conv3)
    # p3 = K.layers.Reshape((int(pool3.shape[1]), int(pool3.shape[3])))(pool3)
    p3 = K.layers.Lambda(squeeze(axis=2))(pool3)

    # [TIMBRE] filter shape 4: 7x0.4f
    conv4 = Conv2D(
        filters=num_filt,
        kernel_size=[7, int(0.4 * input_shape[1])],
        padding="valid",
        activation='relu',
        kernel_initializer=ker_init)(input_pad_7)

    bn_conv4 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv4)
    pool4 = K.layers.MaxPooling2D(pool_size=[1, int(bn_conv4.shape[2])],
                                  strides=[1, int(bn_conv4.shape[2])])(bn_conv4)
    # p4 = K.layers.Reshape((int(pool4.shape[1]), int(pool4.shape[3])))(pool4)
    p4 = K.layers.Lambda(squeeze(axis=2))(pool4)

    # [TIMBRE] filter shape 5: 3x0.4f
    conv5 = Conv2D(
        filters=num_filt * 2,
        kernel_size=[3, int(0.4 * input_shape[1])],
        padding="valid",
        activation='relu',
        kernel_initializer=ker_init)(input_pad_3)

    bn_conv5 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv5)
    pool5 = K.layers.MaxPooling2D(pool_size=[1, int(bn_conv5.shape[2])],
                                  strides=[1, int(bn_conv5.shape[2])])(bn_conv5)
    # p5 = K.layers.Reshape((int(pool5.shape[1]), int(pool5.shape[3])))(pool5)
    p5 = K.layers.Lambda(squeeze(axis=2))(pool5)

    # [TIMBRE] filter shape 6: 1x0.4f
    conv6 = Conv2D(
        filters=num_filt * 4,
        kernel_size=[1, int(0.4 * input_shape[1])],
        padding="valid",
        activation='relu',
        kernel_initializer=ker_init)(aud_input)

    bn_conv6 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv6)
    pool6 = K.layers.MaxPooling2D(pool_size=[1, int(bn_conv6.shape[2])],
                                  strides=[1, int(bn_conv6.shape[2])])(bn_conv6)
    # p6 = K.layers.Reshape((int(pool6.shape[1]), int(pool6.shape[3])))(pool6)
    p6 = K.layers.Lambda(squeeze(axis=2))(pool6)

    # [TEMPORAL-FEATURES] - average pooling + filter shape 7: 165x1
    pool7 = K.layers.AveragePooling2D(pool_size=[1, int(input_shape[1])],
                                      strides=[1, int(input_shape[1])])(aud_input)
    # pool7_rs = K.layers.Reshape((int(pool7.shape[1]), 1))(pool7)
    pool7_rs = K.layers.Lambda(squeeze(axis=2))(pool7)
    conv7 = K.layers.Conv1D(filters=num_filt,
                            kernel_size=165*2,
                            padding="same",
                            activation='relu',
                            kernel_initializer=ker_init)(pool7_rs)
    bn_conv7 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv7)

    # [TEMPORAL-FEATURES] - average pooling + filter shape 8: 128x1
    pool8 = K.layers.AveragePooling2D(pool_size=[1, int(input_shape[1])],
                                      strides=[1, int(input_shape[1])])(aud_input)
    # pool8_rs = K.layers.Reshape((int(pool8.shape[1]), 1))(pool8)
    pool8_rs = K.layers.Lambda(squeeze(axis=2))(pool8)
    conv8 = K.layers.Conv1D(filters=num_filt * 2,
                            kernel_size=128*2,
                            padding="same",
                            activation='relu',
                            kernel_initializer=ker_init)(pool8_rs)
    bn_conv8 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv8)

    # [TEMPORAL-FEATURES] - average pooling + filter shape 9: 64x1
    pool9 = K.layers.AveragePooling2D(pool_size=[1, int(input_shape[1])],
                                      strides=[1, int(input_shape[1])])(aud_input)
    # pool9_rs = K.layers.Reshape((int(pool9.shape[1]), 1))(pool9)
    pool9_rs = K.layers.Lambda(squeeze(axis=2))(pool9)
    conv9 = K.layers.Conv1D(filters=num_filt * 4,
                            kernel_size=64*2,
                            padding="same",
                            activation='relu',
                            kernel_initializer=ker_init)(pool9_rs)
    bn_conv9 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv9)

    # [TEMPORAL-FEATURES] - average pooling + filter shape 10: 32x1
    pool10 = K.layers.AveragePooling2D(pool_size=[1, int(input_shape[1])],
                                       strides=[1, int(input_shape[1])])(aud_input)
    # pool10_rs = K.layers.Reshape((int(pool10.shape[1]), 1))(pool10)
    pool10_rs = K.layers.Lambda(squeeze(axis=2))(pool10)
    conv10 = K.layers.Conv1D(filters=num_filt * 8,
                             kernel_size=32*2,
                             padding="same",
                             activation='relu',
                             kernel_initializer=ker_init)(pool10_rs)
    bn_conv10 = K.layers.BatchNormalization(momentum=0.99,
                                            epsilon=1e-3)(conv10)

    # concatenate all feature maps
    x = K.layers.Concatenate(axis=2)([p1, p2, p3, p4, p5, p6, bn_conv7, bn_conv8, bn_conv9,
                                      bn_conv10])

    x = K.layers.Lambda(expand_dims(axis=2), name='embedding')(x)
    # x = K.layers.Reshape((1, int(x.shape[1]), int(x.shape[2])))(x)
    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='pooling')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='pooling')(x)

    if include_top:
        # FC block
        x = Flatten(name='flatten_')(x)
        x = Dense(4096, activation='relu', name='jordi/fc1_1')(x)
        x = Dense(4096, activation='relu', name='jordi/fc1_2')(x)
        x = Dense(out_dim, activation='relu', name='jordi_fc2')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = aud_input
    # Create model.
    model = Model(inputs, x, name='JoidiPons')

    # load weights
    if load_weights:
        if weights == 'audioset':
            model.load_weights(weights_path)
        else:
            print("failed to load weights")

    return model


def create_jordi_pons(vggish_params, frontend, base_dir, num_classes, input_tensor=None):
    jordi = JordiPons(vggish_params, load_weights=False, include_top=False, pooling='',
                      input_shape=(None, vggish_params.NUM_BANDS, 1),
                      input_tensor=input_tensor, num_filt=16)
    jordi_out = jordi.get_layer('embedding').output
    x = frontend(num_classes)(jordi_out)
    model = Model(jordi.inputs, x, name='model')
    return model
