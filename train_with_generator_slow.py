import argparse
import os
import tensorflow as tf
import keras as K
from keras import Model
from keras.layers import Dense, Softmax

from spec_based.spec_generator import SpecGenerator
from test import auc_roc, auc_pr
from train_with_caching import fronted_from_str
from utils.keras_eval import BestModelSaver
from ast import literal_eval as make_tuple
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.backend.set_session(tf.Session(config=config))

import numpy as np
from librosa.display import specshow
import matplotlib.pyplot as plt
from matplotlib import cm

from vggish.mel_features_np import wavfile_to_examples
from vggish.vggish_arch import VGGish, create_vggish
from vggish.vggish_params import SpecParams

file_path = os.path.dirname(__file__)
BATCH_SIZE = 32


def train_with_generator(model_factory, generator_factory):
    vggish_params = SpecParams()
    vggish_params.EXAMPLE_WINDOW_SECONDS = None

    parser = argparse.ArgumentParser(description='Train')

    parser.add_argument('--dataset', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/train')
    parser.add_argument('--dataset-test', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/test')
    parser.add_argument('--frontend', type=str, default='lstm')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seg', type=str, default='[(0.15, 0.4, 15), (0.4, 0.7, 15)]')
    parser.add_argument('--cachedir', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/cache')
    parser.add_argument('--savedir', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/results/pretr_vgg_(0.15, 0.4, 15), (0.4, 0.7, 15)_rnn_')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    metrics = ['accuracy', auc_roc, auc_pr]
    args = parser.parse_args()

    frontend = fronted_from_str(args.frontend)
    seg = make_tuple(args.seg)
    generator = generator_factory(args.dataset, args.cachedir, vggish_params,
                                  seg,
                                  batch_size=args.batch, val_to_train=0.1, file_lock=False, proc_lock = False)

    model = model_factory(vggish_params, frontend, file_path, len(generator.classes_elements_num))
    optimizer = K.optimizers.Adam(lr=args.lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
    model.summary()

    evaluator = BestModelSaver(generator.make_val_generator(), save_path=args.savedir)
    try:
        model.fit_generator(
            generator.make_train_generator(),
            epochs=200,
            verbose=1,
            callbacks=[
                evaluator
            ],
            use_multiprocessing=False, workers=4,
            class_weight=None)
    except KeyboardInterrupt:
        print('interrupt')
        pass

    model_best = evaluator.best_model
    generator_test = generator_factory(args.dataset_test, args.cachedir, vggish_params,
                                       seg,
                                       batch_size=args.batch, val_to_train=1., file_lock=False, proc_lock = False)
    generator_test_gen = generator_test.make_val_generator()
    model_best.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
    results = model_best.evaluate_generator(generator_test_gen,
                                       verbose=1, use_multiprocessing=False, workers=4)
    metrics_str = '\n'
    for result, name in zip(results, model.metrics_names):
        metric_name = 'test' + '_' + name
        metrics_str = metrics_str + metric_name + ': ' + str(result) + '\n'

    print(metrics_str)

    history = model.history

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
