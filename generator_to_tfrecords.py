import argparse
import functools
import operator
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from logger import get_logger
from spec_based.spec_generator import SpecGenerator
from vggish.vggish_params import SpecParams

log = get_logger('to_tf')


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def cache_generator_to_tfrecord(generator, path, passes=1):
    shape_x, shape_y = None, None
    cnt = 0
    with tf.python_io.TFRecordWriter(path) as writer:
        log.info('make %s' % path)
        for _ in range(passes):
            for i in tqdm(range(len(generator))):
                labels_x, labels_y = generator[i]
                labels_x = labels_x.astype(np.float16)
                if i == 0:
                    shape_x = labels_x.shape[1:]
                    shape_y = labels_y.shape[1:]
                for label_x, label_y in zip(labels_x, labels_y):
                    feature = {'item': _bytes_feature(tf.compat.as_bytes(label_x.tostring())),
                               'label': _float_list_feature(label_y.tolist())}

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    ex = example.SerializeToString()
                    writer.write(ex)
                    cnt += 1

    return shape_x, shape_y, cnt


def parse_function(shape_y):
    def _parse_function(proto):
        keys_to_features = {'item': tf.FixedLenFeature([], tf.string),
                            "label": tf.FixedLenFeature([functools.reduce(operator.mul, shape_y, 1)], tf.float32)}

        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)

        parsed_features['item'] = tf.decode_raw(
            parsed_features['item'], tf.float16)
        parsed_features['item'] = tf.cast(parsed_features['item'], tf.float32)

        return parsed_features['item'], parsed_features['label']

    return _parse_function


def create_dataset(filepath, shape_x, shape_y, nmbr_els, batch_size=64):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(parse_function(shape_y), num_parallel_calls=8)
    dataset = dataset.repeat()
    # dataset = dataset.shuffle(nmbr_els)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    iterator = dataset.make_one_shot_iterator()
    item, label = iterator.get_next()
    shape = [-1]
    shape.extend(list(shape_x))
    item = tf.reshape(item, shape)

    return item, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test caching')

    parser.add_argument('--dataset', type=str, default='H:\work\live_studio\samples')
    parser.add_argument('--cachedir', type=str, default='H:\work\live_studio\cache')
    args = parser.parse_args()
    vggish_params = SpecParams()
    vggish_params.EXAMPLE_WINDOW_SECONDS = None
    generator = SpecGenerator(args.dataset, args.cachedir, vggish_params,
                              [(0.08, 0.3, 6), (0.4, 0.6, 6), (0.7, 0.92, 6)],
                              batch_size=64, val_to_train=0.15, file_lock=False)
    shape_x, shape_y, cnt = cache_generator_to_tfrecord(generator.make_train_generator(), 'F:\\tmp\\cache.tf')

    item, label = create_dataset('F:\\tmp\\cache.tf', shape_x, shape_y, cnt)

    with tf.Session() as sess:
        v = sess.run(label)
        print(v)
