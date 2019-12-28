from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
import keras as K
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, \
    Softmax, AveragePooling2D, Conv1D
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


def WaveArch(vggish_params, load_weights=True, weights='audioset',
             input_tensor=None, input_shape=None, include_top=False,
             out_dim=None, weights_path=None, pooling='avg'):
    '''
    An implementation of the wave architecture.

    :param load_weights: if load weights
    :param weights: loads weights pre-trained on a preliminary version of YouTube-8M.
    :param input_tensor: input_layer
    :param input_shape: input data shape
    :param out_dim: output dimension


    :return: A Keras model instance.
    '''

    if weights not in {'audioset', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `audioset` '
                         '(pre-training on audioset).')

    if out_dim is None:
        out_dim = vggish_params.EMBEDDING_SIZE

    # input shape
    if input_shape is None:
        input_shape = (None, 1)

    if input_tensor is None:
        aud_input = Input(shape=input_shape, name='input_1')
    else:
        if not K.backend.is_keras_tensor(input_tensor):
            aud_input = Input(tensor=input_tensor, shape=input_shape, name='input_1')
        else:
            aud_input = input_tensor

    init = K.initializers.VarianceScaling()

    conv0 = Conv1D(filters=64,
                   kernel_size=3,
                   strides=3,
                   padding="valid",
                   activation='relu',
                   kernel_initializer=init)(aud_input)
    bn_conv0 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv0)

    conv1 = Conv1D(filters=64,
                   kernel_size=3,
                   strides=1,
                   padding="valid",
                   activation='relu',
                   kernel_initializer=init)(bn_conv0)
    bn_conv1 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv1)
    pool_1 = K.layers.MaxPooling1D(pool_size=3, strides=3)(bn_conv1)

    conv2 = Conv1D(filters=64,
                   kernel_size=3,
                   strides=1,
                   padding="valid",
                   activation='relu',
                   kernel_initializer=init)(pool_1)
    bn_conv2 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv2)
    pool_2 = K.layers.MaxPooling1D(pool_size=3, strides=3)(bn_conv2)

    conv3 = Conv1D(filters=128,
                   kernel_size=3,
                   strides=1,
                   padding="valid",
                   activation='relu',
                   kernel_initializer=init)(pool_2)
    bn_conv3 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv3)
    pool_3 = K.layers.MaxPooling1D(pool_size=3, strides=3)(bn_conv3)

    conv4 = Conv1D(filters=128,
                   kernel_size=3,
                   strides=1,
                   padding="valid",
                   activation='relu',
                   kernel_initializer=init)(pool_3)
    bn_conv4 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv4)
    pool_4 = K.layers.MaxPooling1D(pool_size=3, strides=3)(bn_conv4)

    conv5 = Conv1D(filters=128,
                   kernel_size=3,
                   strides=1,
                   padding="valid",
                   activation='relu',
                   kernel_initializer=init)(pool_4)
    bn_conv5 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv5)
    pool_5 = K.layers.MaxPooling1D(pool_size=3, strides=3)(bn_conv5)

    conv6 = Conv1D(filters=256,
                   kernel_size=3,
                   strides=1,
                   padding="valid",
                   activation='relu',
                   kernel_initializer=init)(pool_5)
    bn_conv6 = K.layers.BatchNormalization(momentum=0.99,
                                           epsilon=1e-3)(conv6)
    x = K.layers.MaxPooling1D(pool_size=3, strides=3)(bn_conv6)

    x = K.layers.Lambda(expand_dims(2), name='embedding')(x)
    # x = K.layers.Reshape((1, int(x.shape[1]), int(x.shape[2])))(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='pooling')(x)
    elif pooling == 'max':
        x = GlobalAveragePooling2D(name='pooling')(x)

    if include_top:
        # FC block
        x = Flatten(name='flatten_')(x)
        x = Dense(4096, activation='relu', name='wave/fc1_1')(x)
        x = Dense(4096, activation='relu', name='wave/fc1_2')(x)
        x = Dense(out_dim, activation='relu', name='wave_fc2')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = aud_input
    # Create model.
    model = Model(inputs, x, name='WaveArch')

    # load weights
    if load_weights:
        if weights == 'audioset':
            model.load_weights(weights_path)
        else:
            print("failed to load weights")

    return model


def create_wave_arch(vggish_params, frontend, base_dir, num_classes, input_tensor=None):
    wavenet = WaveArch(vggish_params, load_weights=False, include_top=False, pooling='',
                       input_shape=(None, 1),
                       input_tensor=input_tensor)
    vggish_out = wavenet.get_layer('embedding').output
    x = frontend(num_classes)(vggish_out)
    model = Model(wavenet.inputs, x, name='model')
    # model.load_weights('/hdd/youtube8m/youtube-8m-videos-frames/h5/weights_.02-0.15-0.94.hdf5')
    return model
