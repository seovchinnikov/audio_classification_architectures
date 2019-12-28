import argparse
import cPickle
import os

import joblib
import tensorflow as tf
import keras as K
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats
from sklearn import svm
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from tensorflow.contrib.signal import mfccs_from_log_mel_spectrograms
import xgboost as xgb
from generator_to_tfrecords import create_dataset, cache_generator_to_tfrecord
from spec_based.spec_generator import SpecGenerator
from utils.keras_eval import EvaluateInputTensor

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.backend.set_session(tf.Session(config=config))
from vggish.vggish_params import SpecParams

file_path = os.path.dirname(__file__)
BATCH_SIZE = 128


def compute_mean_var(train_items, cnt, batch_size=1280):
    X = []
    transformer = StandardScaler()
    train_items = tf.reshape(train_items, (-1, tf.shape(train_items)[1] * tf.shape(train_items)[2]))
    with tf.Session() as sess:
        for i in range(cnt):
            items = sess.run(train_items)
            X.extend(list(items))
            if len(X) > batch_size:
                X = np.asarray(X)
                transformer.partial_fit(X)
                X = []

        if len(X) >= 1:
            X = np.asarray(X)
            transformer.partial_fit(X)

    return transformer.mean_, transformer.scale_

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def fit_pca(train_items, cnt, batch_size=1280, n_comp=128):
    mean, var = 0., 1.# compute_mean_var(train_items, cnt, batch_size=batch_size)

    transformer = IncrementalPCA(n_components=n_comp, batch_size=batch_size, whiten=False)
    transformer.mean_my = mean
    transformer.var_my = var
    train_items = tf.reshape(train_items, (-1, tf.shape(train_items)[1] * tf.shape(train_items)[2]))
    X = []
    with tf.Session() as sess:
        for i in range(cnt):
            items = sess.run(train_items)
            X.extend(list(items))
            if len(X) > batch_size:
                X = np.asarray(X)
                X = (X - mean) / var
                transformer.partial_fit(X)
                X = []

        if len(X) >= n_comp:
            X = np.asarray(X)
            X = (X - mean) / var
            transformer.partial_fit(X)

    print('explained %s' % transformer.explained_variance_ratio_)
    return transformer


def transform_pca(tensor, transformer):
    tensor = tf.reshape(tensor, (-1, tf.shape(tensor)[1] * tf.shape(tensor)[2]))
    if transformer.mean_ is not None:
        mean_my = tf.convert_to_tensor(transformer.mean_my, dtype=tf.float32)
        var_my = tf.convert_to_tensor(transformer.var_my, dtype=tf.float32)
        tensor = (tensor - mean_my) / var_my
        # mean = tf.reshape(mean, (1, tensor.shape[1], tensor.shape[2]))
        # np.reshape(transformer.mean_, (1, int(tensor.shape[1]), int(tensor.shape[2])))
        mean = tf.convert_to_tensor(transformer.mean_, dtype=tf.float32)
        tensor = tensor - mean

    transf = tf.convert_to_tensor(transformer.components_.T, dtype=tf.float32)
    # transf = tf.reshape(transf, (1, tensor.shape[1], tensor.shape[2]))
    X_transformed = tf.matmul(tensor, transf)
    # print(X_transformed.shape)
    return X_transformed


def prepare_data(x_transformed, labels, cnt):
    X = []
    Y = []
    with tf.Session() as sess:
        for i in range(cnt):
            items_x, items_y = sess.run([x_transformed, labels])
            X.extend(list(items_x))
            items_y = np.argmax(items_y, axis=1)
            Y.extend(list(items_y))
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y


def random_crop_time(tensor, time):
    # return tensor
    tensor = tf.random_crop(tensor, [
        tf.shape(tensor)[0], int(time), tf.shape(tensor)[2]])
    # tensor = tf.reshape(tensor, [tf.shape(tensor)[0], -1])
    return tensor


def prepate_data_for_voting(sess, ipca, tensor, labels, times, interval_len):
    t_res_x, t_res_y = sess.run([tensor, labels])
    res_x = []
    res_y = []
    interval_len = int(interval_len)
    for j in range(t_res_x.shape[0]):
        res_j = []
        t_res_j = t_res_x[j]
        int_start = 0
        for i in range(times):
            res_j.append(t_res_j[int_start:int_start + interval_len])
            int_start += interval_len

        res_x.append(res_j)
        res_y.append(t_res_y[j])

    return np.asarray(res_x), np.asarray(res_y)


def test_voting(model, ipca, tensor, labels, times, interval_len, cnt):
    ok = 0.
    total = 0.
    with tf.Session() as sess:
        for i in range(cnt):
            x, y = prepate_data_for_voting(sess, ipca, tensor, labels, times, interval_len)
            num, width, temp, feat = x.shape
            x = np.reshape(x, (-1, x.shape[2] * x.shape[3]))

            x = (x - ipca.mean_my) / ipca.var_my
            x = ipca.transform(x)
            y_pred = model.predict(x)
            y_pred = np.reshape(y_pred, (-1, width))
            y_pred = stats.mode(y_pred, axis=1)[0]
            y_pred = np.reshape(y_pred, (-1))
            y = np.argmax(y, axis=1)
            ok += np.sum(y == y_pred, axis=0)
            total += y.shape[0]

    return ok / total


def classifier_factory(name):
    if name == 'xgb':
        clf_xgb = xgb.XGBClassifier(objective='binary:logistic'
                                    # 'binary:logistic'
                                    )
        param_dist = {'n_estimators': [60, 90, 100, 120],
                      'learning_rate': [0.01, 0.05, 0.1, 0.2],
                      'subsample': [0.7, 0.9, 1.],
                      'max_depth': [9, 11, 13, 15, 17],
                      'colsample_bytree': [0.8, 0.9, 1.0],
                      'min_child_weight': [0.5, 1, 2, 3],
                      'nthread': [4]
                      }
        fit_params = {
            # 'eval_set': [(X_test, y_test)]
        }
        return clf_xgb, param_dist, fit_params

    elif name == 'svm':
        clf = svm.SVC()
        param_dist = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
        # param_dist = [
        #     {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [1, 2, 3, 4]},
        #     {'C': [0.1, 1, 10, 100, 1000], 'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
        # ]
        fit_params = {
            # 'eval_set': [(X_test, y_test)]
        }

        return clf, param_dist, fit_params

    elif name == 'lsvm':
        clf = LinearSVC(dual=False, verbose=0)
        param_dist = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
        # param_dist = [
        #     {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [1, 2, 3, 4]},
        #     {'C': [0.1, 1, 10, 100, 1000], 'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
        # ]
        fit_params = {
            # 'eval_set': [(X_test, y_test)]
        }

        return clf, param_dist, fit_params


def train_with_cache(generator_factory):
    sess = K.backend.get_session()

    vggish_params = SpecParams()
    vggish_params.EXAMPLE_WINDOW_SECONDS = None

    parser = argparse.ArgumentParser(description='Train')

    parser.add_argument('--dataset', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/train')
    parser.add_argument('--ds-passes', type=int, default=3)
    parser.add_argument('--dataset-test', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/test')
    parser.add_argument('--cachedir', type=str, default='/hdd/youtube8m/youtube-8m-videos-frames/cache')
    parser.add_argument('--savedir', type=str,
                        default='/hdd/youtube8m/youtube-8m-videos-frames/results/temp')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--classifier', type=str, default='xgb')
    parser.add_argument('--voting', type=int, default=0)
    parser.add_argument('--votes', type=int, default=9)
    parser.add_argument('--samples-from-pass', type=int, default=3)

    args = parser.parse_args()

    generator_train = generator_factory(args.dataset, args.cachedir, vggish_params,
                                        [(0.15, 0.4, 15), (0.4, 0.7, 15)],
                                        batch_size=args.batch, val_to_train=0.001, file_lock=False)
    generator_test = generator_factory(args.dataset_test, args.cachedir, vggish_params,
                                       [(0.15, 0.4, 15), (0.4, 0.7, 15)],
                                       batch_size=args.batch, val_to_train=1., file_lock=False)

    tf_rec_train = os.path.join(args.cachedir,
                                generator_train.obtain_params_descriptor(vggish_params) + '_train_mfcc.tfrec')

    tf_rec_val = os.path.join(args.cachedir,
                              generator_train.obtain_params_descriptor(vggish_params) + '_val_mfcc.tfrec')

    tf_rec_test = os.path.join(args.cachedir,
                               generator_test.obtain_params_descriptor(vggish_params) + '_test_mfcc.tfrec')

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
    train_items = mfccs_from_log_mel_spectrograms(tf.squeeze(train_items, axis=-1))[..., :13]
    val_items, val_labels = create_dataset(tf_rec_val, shape_val_x, shape_val_y, cnt_val, batch_size=args.batch)
    val_items = mfccs_from_log_mel_spectrograms(tf.squeeze(val_items, axis=-1))[..., :13]
    test_items, test_labels = create_dataset(tf_rec_test, shape_test_x, shape_test_y, cnt_test, batch_size=args.batch)
    test_items = mfccs_from_log_mel_spectrograms(tf.squeeze(test_items, axis=-1))[..., :13]

    samples_from_pass = args.samples_from_pass
    if not args.voting or args.voting <= 0:
        ipca = fit_pca(train_items, int(np.ceil(float(cnt_train) / args.batch)), batch_size=512)
        train_items = transform_pca(train_items, ipca)
        val_items = transform_pca(val_items, ipca)
        test_items = transform_pca(test_items, ipca)
        samples_from_pass = 1
    else:
        train_items = random_crop_time(train_items, int(args.voting / vggish_params.STFT_HOP_LENGTH_SECONDS))
        val_items = random_crop_time(val_items, int(args.voting / vggish_params.STFT_HOP_LENGTH_SECONDS))
        ipca = fit_pca(train_items, int(np.ceil(float(cnt_train) / args.batch)), 512)
        train_items = transform_pca(train_items, ipca)
        val_items = transform_pca(val_items, ipca)
        # test_items = random_crop_time(test_items, int(args.voting / vggish_params.STFT_HOP_LENGTH_SECONDS))

    X_train, Y_train = prepare_data(train_items, train_labels,
                                    samples_from_pass * int(np.ceil(float(cnt_train) / args.batch)))
    X_val, Y_val = prepare_data(val_items, val_labels, int(np.ceil(float(cnt_train) / args.batch)))

    clf, param_dist, fit_params = classifier_factory(args.classifier)

    clf = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, fit_params=fit_params, cv=6,
                             refit=True, scoring='accuracy', verbose=10, n_jobs=-1)
    clf.fit(X_train, Y_train)

    best_score = clf.best_score_
    best_params = clf.best_params_
    print("Best mean accuracy: {}".format(best_score))
    print("Best params: ")
    for param_name in sorted(best_params.keys()):
        print('%s: %r' % (param_name, best_params[param_name]))

    bst = clf.best_estimator_

    if not args.voting or args.voting <= 0:
        X_test, Y_test = prepare_data(test_items, test_labels, int(np.ceil(float(cnt_train) / args.batch)))
        y_pred = bst.predict(X_test)
        print('avg accuracy on test set', accuracy_score(Y_test, y_pred))
        y_pred_proba = bst.predict_proba(X_test)
        print('avg average_precision_score on test set', average_precision_score(Y_test, y_pred_proba[:, 1]))
    else:
        acc = test_voting(bst, ipca, test_items, test_labels, args.votes,
                          args.voting / vggish_params.STFT_HOP_LENGTH_SECONDS,
                          int(np.ceil(float(cnt_test) / args.batch)))
        print('avg accuracy on test set', acc)

    with open(os.path.join(args.savedir, 'model.pkl'), 'wb') as fid:
        joblib.dump(bst, fid)

    with open(os.path.join(args.savedir, 'ipca.pkl'), 'wb') as fid:
        joblib.dump(ipca, fid)


train_with_cache(SpecGenerator)
