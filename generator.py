import glob
import math
import multiprocessing
import random
import threading
import time

import keras as K
import os

import numpy as np
import cv2
from filelock import FileLock
from keras.utils import Sequence
import logging
from sklearn.utils.class_weight import compute_class_weight

from logger import get_logger
from vggish.mel_features_np import wavfile_to_examples

logger = get_logger('generator')

file_path = os.path.dirname(__file__)
SEPARATOR = os.path.sep


class BaseGenerator:
    def __init__(self, ds_folder, cache_dir, spec_params, sampling_itvs, batch_size=64, val_to_train=0.15,
                 preprocessor=None, file_lock=False, proc_lock=True):
        cats_folders = list(sorted([name for name in os.listdir(ds_folder)
                                    if os.path.isdir(os.path.join(ds_folder, name))]))
        self.cache_dir = cache_dir
        self.spec_params = spec_params
        self.sampling_itvs = sampling_itvs
        self.random_state = np.random.RandomState(1234)
        self.ds_items = []
        self.ds_labels = []
        self.name_to_idx = {}
        self.idx_to_name = {}
        self.file_lock = file_lock

        logger.debug("start init of dataset")

        self.classes_elements_num = {}
        self.max_classes_elements_num = 0

        self.locks = {}
        for i, cat in enumerate(cats_folders):
            cnt = 0
            if not os.path.exists(os.path.join(self.cache_dir, str(i))):
                os.makedirs(os.path.join(self.cache_dir, str(i)))

            for item_path in self.ls_dir(os.path.join(ds_folder, str(cat))):
                self.ds_items.append(item_path)
                self.ds_labels.append(i)
                cnt += 1
                if not self.file_lock:
                    self.locks[item_path] = multiprocessing.Lock() if proc_lock else threading.Lock()

            self.max_classes_elements_num = max([self.max_classes_elements_num, cnt])
            self.name_to_idx[cat] = i
            self.idx_to_name[i] = cat
            self.classes_elements_num[i] = cnt

        self.ds_items = np.asarray(self.ds_items)
        self.ds_labels = np.asarray(self.ds_labels)
        self.all_length = self.ds_items.shape[0]
        self.batch_size = batch_size
        self.h = self.spec_params.NUM_BANDS
        self.val_to_train = val_to_train
        self.cats_num = len(cats_folders)

        self.preprocessor = preprocessor

        all_indices = np.arange(self.ds_items.shape[0])
        self.random_state.shuffle(all_indices)
        self.train_indices, self.val_indices = np.split(all_indices,
                                                        [int((1 - val_to_train) * self.all_length)])

        self.train_length = self.train_indices.shape[0]
        self.val_length = self.val_indices.shape[0]
        logger.debug("dataset inited")
        logger.debug("total %s items in train", self.train_length)
        logger.debug("total %s items in val", self.val_length)
        logger.debug("total %s categories", self.cats_num)

    @staticmethod
    def ls_dir(dir):
        return glob.glob(dir + SEPARATOR + '*.wav') + \
               glob.glob(dir + SEPARATOR + '*.ogg') + \
               glob.glob(dir + SEPARATOR + '*.mp3') + \
               glob.glob(dir + SEPARATOR + '*.opus') + glob.glob(dir + SEPARATOR + '*.m4a') + \
               glob.glob(dir + SEPARATOR + '*.aac') + glob.glob(dir + SEPARATOR + '*.webm')

    def obtain_params_descriptor(self, spec_params):
        raise Exception('unimplemented')

    def obtain_class_weights(self):
        # res = {}
        # for item, val in self.classes_elements_num.items():
        #     res[item] = float(self.max_classes_elements_num) / (val + 1)
        #
        class_weights = compute_class_weight('balanced', np.array(range(self.cats_num)), self.ds_labels)
        res = dict(enumerate(class_weights))
        return res

    def obtain_temp_length(self):
        res = 0
        for v in self.sampling_itvs:
            res += int(float(v[2]) / self.spec_params.STFT_HOP_LENGTH_SECONDS)

        return res

    class BaseInnerGenerator(Sequence):

        def __init__(self, outer, indexes, is_train):
            self.indixes = indexes
            self.outer = outer
            self.length = self.indixes.shape[0]
            self.is_train = is_train

        def on_epoch_end(self):
            self.outer.random_state.shuffle(self.indixes)

        def __getitem__(self, idx):

            inds = self.indixes[idx * self.outer.batch_size:(idx + 1) * self.outer.batch_size]
            items_path = self.outer.ds_items[inds]
            labels = self.outer.ds_labels[inds]

            labels_y = []
            items = []
            for i in range(inds.shape[0]):
                try:
                    file_name = os.path.basename(items_path[i])
                    cache_file = os.path.join(self.outer.cache_dir, str(labels[i]),
                                              file_name + 'x' + self.outer.obtain_params_descriptor(
                                                  self.outer.spec_params) + '.npy')
                    with self.lock(items_path[i]):
                        if os.path.exists(cache_file):
                            content = np.load(cache_file, mmap_mode='r')
                        else:
                            content = self.read_audio(items_path[i])
                            np.save(cache_file, content.astype(np.float16))

                    content = self.preproc_audio(content).astype(np.float32)
                    items.append(content)
                    labels_y.append(K.utils.to_categorical([int(labels[i])], num_classes=self.outer.cats_num)[0])
                except Exception as e:
                    logger.exception("error on generation")

            items = np.array(items)

            if self.outer.preprocessor is not None:
                items = self.outer.preprocessor(items)

            labels_x = items
            labels_y = np.asarray(labels_y)

            return labels_x, labels_y

        def __len__(self):
            return int(math.ceil(float(self.length) / self.outer.batch_size))

        def preproc_audio(self, specto):
            raise Exception('unimplemented')

        def read_audio(self, file_name):
            raise Exception('unimplemented')

        def lock(self, file):
            if self.outer.file_lock:
                lock = FileLock(file + ".lock", timeout=25)
                return lock.acquire(25, poll_intervall=0.05)
            else:
                return self.outer.locks[file]

    def make_train_generator(self):
        raise Exception('unimplemented')

    def make_val_generator(self):
        raise Exception('unimplemented')
