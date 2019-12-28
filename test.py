import argparse
import os
import tensorflow as tf
import keras as K
import time
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval as make_tuple
from frontends import feature_maxpooling, feature_avgpooling, decision_avgpooling, feature_avgmaxpooling, \
    decision_attention, jordi_frontend, lstm
from generator_to_tfrecords import create_dataset, cache_generator_to_tfrecord
from logger import get_logger
from spec_based.spec_generator import SpecGenerator
from utils.keras_eval import EvaluateInputTensor
from wave_based.wave_generator import WaveGenerator

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.backend.set_session(tf.Session(config=config))

from vggish.vggish_params import SpecParams

file_path = os.path.dirname(__file__)
BATCH_SIZE = 32
logger = get_logger('test')


def fronted_from_str(name):
    if name == 'feature_maxpooling':
        return feature_maxpooling
    elif name == 'feature_avgpooling':
        return feature_avgpooling
    elif name == 'feature_avgmaxpooling':
        return feature_avgmaxpooling
    elif name == 'decision_avgpooling':
        return decision_avgpooling
    elif name == 'decision_attention':
        return decision_attention
    elif name == 'jordi_frontend':
        return jordi_frontend
    elif name == 'lstm':
        return lstm


def binary_acc(y_true, y_pred):
    # value, update_op = tf.metrics.precision_at_thresholds(
    #     y_true,
    #     y_pred,
    #     np.arange(0, 1., 0.01, dtype=np.float32),
    #     name='pratt')
    #
    # # find all variables created for this metric
    # metric_vars = [i for i in tf.local_variables() if 'pratt' in i.name.split('/')[2]]
    #
    # # Add metric variables to GLOBAL_VARIABLES collection.
    # # They will be initialized for new session.
    # for v in metric_vars:
    #     tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
    #
    # # force to update metric values
    # with tf.control_dependencies([update_op]):
    #     value = tf.identity(value)
    #     return tf.reduce_max(value)
    t = float(0)
    for v in np.arange(0, 1., 0.01, dtype=np.float32):
        c = K.backend.mean(K.backend.cast(
            K.backend.equal(y_true[:, 0], K.backend.cast(K.backend.less(v, y_pred[:, 0]), dtype=tf.float32)),
            dtype=tf.float32))
        t = tf.maximum(t, c)

    return t


def auc_pr(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred, name='auc_pr', curve='PR',
                                      summation_method='careful_interpolation')

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_pr' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred, curve='ROC', name='auc_roc',
                                      summation_method='careful_interpolation')

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def test_with_cache(generator_factory):
    sess = K.backend.get_session()

    vggish_params = SpecParams()
    vggish_params.EXAMPLE_WINDOW_SECONDS = None

    parser = argparse.ArgumentParser(description='Train')

    parser.add_argument('--ds-passes', type=int, default=7)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seg', type=str, default='[(0.15, 0.4, 15), (0.4, 0.7, 15)]')
    parser.add_argument('--dataset-test', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/test')
    parser.add_argument('--cachedir', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/cache')
    parser.add_argument('--savedir', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/results/temp')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--model', type=str,
                        default='/hdd/youtube8m/youtube-8m-videos-frames/results_map/awared_jordi_(0.15,0.4,15), (0.4, 0.7, 15)_81.7/w_15-acc_0.9419_loss_0.159_val_auc_pr_0.9628_val_auc_roc_0.9639_auc_roc_0.9698_val_acc_0.9307_auc_pr_0.9687_val_loss_0.1965_.hdf5')
    args = parser.parse_args()
    seg = make_tuple(args.seg)

    generator_test = generator_factory(args.dataset_test, args.cachedir, vggish_params,
                                       seg,
                                       batch_size=args.batch, val_to_train=1., file_lock=False)

    tf_rec_test = os.path.join(args.cachedir,
                               generator_test.obtain_params_descriptor(vggish_params) + '_test.tfrec')

    test_gen = generator_test.make_val_generator()
    shape_test_x, shape_test_y, cnt_test = cache_generator_to_tfrecord(test_gen,
                                                                       tf_rec_test, args.ds_passes)

    test_items, test_labels = create_dataset(tf_rec_test, shape_test_x, shape_test_y, cnt_test, batch_size=args.batch)
    test_model_input = K.Input(tensor=test_items)
    model_test = K.models.load_model(args.model, custom_objects={'auc_roc':auc_roc, 'auc_pr':auc_pr, 'binary_acc': binary_acc})
    model_test.layers.pop(0)
    test_model_out = model_test(test_model_input)
    model_test = K.models.Model(test_model_input, test_model_out)
    optimizer = K.optimizers.Adam(lr=args.lr)
    model_test.compile(optimizer=optimizer, loss='categorical_crossentropy',
                       metrics=['accuracy', auc_roc, auc_pr, binary_acc],
                       target_tensors=test_labels)

    _ = model_test.evaluate(None, None, steps=1,
                                  verbose=1)
    start = time.time()
    results = model_test.evaluate(None, None, steps=int(np.ceil(cnt_test / float(args.batch))),
                                  verbose=1)
    logger.info('%s items per sec', int(args.batch * np.ceil(cnt_test / float(args.batch))) / (time.time() - start))

    metrics_str = '\n'
    for result, name in zip(results, model_test.metrics_names):
        metric_name = 'test' + '_' + name
        metrics_str = metrics_str + metric_name + ': ' + str(result) + '\n'

    print(metrics_str)

    # model_train.save_weights(os.path.join(file_path, '..', 'h5'), "weights_.{epoch:02d}-{val_loss:.2f}-{acc:.2f}.hdf5")

    K.backend.clear_session()


if __name__ == '__main__':
    test_with_cache(SpecGenerator)
