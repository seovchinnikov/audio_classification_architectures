import argparse
import os
import tensorflow as tf
import keras as K
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval as make_tuple
from frontends import feature_maxpooling, feature_avgpooling, decision_avgpooling, feature_avgmaxpooling, \
    decision_attention, jordi_frontend, lstm
from generator_to_tfrecords import create_dataset, cache_generator_to_tfrecord
from spec_based.spec_generator import SpecGenerator
from test import auc_roc, auc_pr
from utils.keras_eval import EvaluateInputTensor

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.backend.set_session(tf.Session(config=config))

from vggish.vggish_params import SpecParams

file_path = os.path.dirname(__file__)
BATCH_SIZE = 32


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


def train_with_cache(model_factory, generator_factory):
    sess = K.backend.get_session()

    vggish_params = SpecParams()
    vggish_params.EXAMPLE_WINDOW_SECONDS = None

    parser = argparse.ArgumentParser(description='Train')

    parser.add_argument('--dataset', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/train')
    parser.add_argument('--ds-passes', type=int, default=3)
    parser.add_argument('--frontend', type=str, default='jordi_frontend')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seg', type=str, default='[(0.15, 0.4, 15), (0.4, 0.7, 15)]')
    parser.add_argument('--dataset-test', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/test')
    parser.add_argument('--cachedir', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/cache')
    parser.add_argument('--savedir', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/results_map/temp')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    metrics = ['accuracy', auc_roc, auc_pr]
    args = parser.parse_args()
    seg = make_tuple(args.seg)
    frontend = fronted_from_str(args.frontend)

    generator_train = generator_factory(args.dataset, args.cachedir, vggish_params,
                                        seg,
                                        batch_size=args.batch, val_to_train=0.1, file_lock=False)
    generator_test = generator_factory(args.dataset_test, args.cachedir, vggish_params,
                                       seg,
                                       batch_size=args.batch, val_to_train=1., file_lock=False)

    tf_rec_train = os.path.join(args.cachedir,
                                generator_train.obtain_params_descriptor(vggish_params) + '_train.tfrec')

    tf_rec_val = os.path.join(args.cachedir,
                              generator_train.obtain_params_descriptor(vggish_params) + '_val.tfrec')

    tf_rec_test = os.path.join(args.cachedir,
                               generator_test.obtain_params_descriptor(vggish_params) + '_test.tfrec')

    train_gen = generator_train.make_train_generator()
    shape_train_x, shape_train_y, cnt_train = cache_generator_to_tfrecord(train_gen,
                                                                          tf_rec_train, args.ds_passes)
    val_gen = generator_train.make_val_generator()
    shape_val_x, shape_val_y, cnt_val = cache_generator_to_tfrecord(val_gen,
                                                                    tf_rec_val, args.ds_passes)

    test_gen = generator_test.make_val_generator()
    shape_test_x, shape_test_y, cnt_test = cache_generator_to_tfrecord(test_gen,
                                                                       tf_rec_test, args.ds_passes)

    train_items, train_labels = create_dataset(tf_rec_train, shape_train_x, shape_train_y, cnt_train,
                                               batch_size=args.batch)
    val_items, val_labels = create_dataset(tf_rec_val, shape_val_x, shape_val_y, cnt_val, batch_size=args.batch)
    test_items, test_labels = create_dataset(tf_rec_test, shape_test_x, shape_test_y, cnt_test, batch_size=args.batch)

    train_model_input = K.Input(tensor=train_items)
    model_train = model_factory(vggish_params, frontend, file_path, len(generator_train.classes_elements_num),
                                input_tensor=train_model_input)
    optimizer = K.optimizers.Adam(lr=args.lr)
    model_train.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics,
                        target_tensors=train_labels)
    model_train.summary()

    val_model_input = K.Input(tensor=val_items)
    model_val = model_factory(vggish_params, frontend, file_path, len(generator_train.classes_elements_num),
                              input_tensor=val_model_input)
    model_val.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics,
                      target_tensors=val_labels)

    # # Fit the model using data from the TFRecord data tensors.
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess, coord)
    evaluator = EvaluateInputTensor(model_val, save_path=args.savedir,
                                    steps=int(np.ceil(cnt_val / float(args.batch))))

    try:
        model_train.fit(
            epochs=100,
            steps_per_epoch=int(np.ceil(cnt_train / float(args.batch))),
            callbacks=[evaluator],
            verbose=2)

    except KeyboardInterrupt:
        pass

    history = model_train.history
    model_to_test_copy = evaluator.best_model
    test_model_input = K.Input(tensor=test_items)
    model_test = model_factory(vggish_params, frontend, file_path, len(generator_test.classes_elements_num),
                               input_tensor=test_model_input)
    model_test.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics,
                       target_tensors=test_labels)
    model_test.set_weights(model_to_test_copy.get_weights())

    results = model_test.evaluate(None, None, steps=int(np.ceil(cnt_test / float(args.batch))),
                                  verbose=1)
    metrics_str = '\n'
    for result, name in zip(results, model_test.metrics_names):
        metric_name = 'test' + '_' + name
        metrics_str = metrics_str + metric_name + ': ' + str(result) + '\n'

    print(metrics_str)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(args.savedir, 'acc.png'))
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(args.savedir, 'loss.png'))
    plt.show()

    # model_train.save_weights(os.path.join(file_path, '..', 'h5'), "weights_.{epoch:02d}-{val_loss:.2f}-{acc:.2f}.hdf5")

    K.backend.clear_session()
