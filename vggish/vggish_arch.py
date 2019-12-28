"""VGGish model for Keras. A VGG-like model for audio classification

# Reference

- [CNN Architectures for Large-Scale Audio Classification](ICASSP 2017)

"""

from __future__ import print_function
from __future__ import absolute_import

import os

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Lambda, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, \
    Softmax
from keras.engine.topology import get_source_inputs
from keras import backend as K

# weight path
from spec_based.spec_utils import spec_aug


def VGGish(vggish_params, load_weights=True, weights='audioset',
           input_tensor=None, input_shape=None,
           out_dim=None, include_top=True, pooling='avg', weights_path=None, aug=None):
    '''
    An implementation of the VGGish architecture.

    :param load_weights: if load weights
    :param weights: loads weights pre-trained on a preliminary version of YouTube-8M.
    :param input_tensor: input_layer
    :param input_shape: input data shape
    :param out_dim: output dimension
    :param include_top:whether to include the 3 fully-connected layers at the top of the network.
    :param pooling: pooling type over the non-top network, 'avg' or 'max'

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
        if vggish_params.EXAMPLE_WINDOW_SECONDS is not None:
            input_shape = (vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1)
        else:
            input_shape = (None, vggish_params.NUM_BANDS, 1)

    if input_tensor is None:
        aud_input = Input(shape=input_shape, name='input_1')
    else:
        if not K.is_keras_tensor(input_tensor):
            aud_input = Input(tensor=input_tensor, shape=input_shape, name='input_1')
        else:
            aud_input = input_tensor

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = aud_input

    x = aud_input
    if aug is not None:
        x = Lambda(aug())(x)

    # Block 1
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='embedding')(x)

    if include_top:
        # FC block
        x = Flatten(name='flatten_')(x)
        x = Dense(4096, activation='relu', name='vggish_fc1/fc1_1')(x)
        x = Dense(4096, activation='relu', name='vggish_fc1/fc1_2')(x)
        x = Dense(out_dim, activation='relu', name='vggish_fc2')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='pooling')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='pooling')(x)

    # Create model.
    model = Model(inputs, x, name='VGGish')

    # load weights
    if load_weights:
        if weights == 'audioset':
            model.load_weights(weights_path)
        else:
            print("failed to load weights")

    return model


def create_vggish(vggish_params, frontend, base_dir, num_classes, input_tensor=None):
    vggish = VGGish(vggish_params, load_weights=True, include_top=False, pooling='',
                    input_shape=(None, vggish_params.NUM_BANDS, 1),
                    weights_path=os.path.join(base_dir, 'h5', 'vggish_audioset_weights_without_fc2.h5'),
                    input_tensor=input_tensor)
    vggish_out = vggish.get_layer('embedding').output
    x = frontend(num_classes)(vggish_out)
    model = Model(vggish.inputs, x, name='model')
    # model.load_weights('/hdd/youtube8m/youtube-8m-videos-frames/h5/weights_.02-0.15-0.94.hdf5')
    return model
