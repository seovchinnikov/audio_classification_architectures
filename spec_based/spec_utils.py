

def spec_aug():
    def f(x):
        import keras as K
        import tensorflow as tf
        from spec_based.spec_aug import random_erasing, crop_pad

        def aug(x):
            import functools
            # res = functools.partial(random_erasing)
            res = functools.reduce(lambda r, f: f(r), (random_erasing, crop_pad), x)
            # res = tf.Print(res, [tf.shape(res)], message="This is a: ")
            return res

        res = tf.cond(K.backend.learning_phase(), lambda: tf.map_fn(aug, x), lambda: x)
        # res = tf.Print(res, [tf.shape(res)], message="This is a: ")
        return res

    return f
