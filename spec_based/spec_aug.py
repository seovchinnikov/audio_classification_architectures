import tensorflow as tf


def replace_slice(input_, replacement, begin, size=None):
    inp_shape = tf.shape(input_)
    if size is None:
        size = tf.shape(replacement)
    else:
        replacement = tf.broadcast_to(replacement, size)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)


def random_erasing(img, probability=0.5, sl=0.02, sh=0.8, r1=0.3, cnt=2, max_w=0.5):
    '''
    img is a 3-D variable (ex: tf.Variable(image, validate_shape=False) ) and  HWC order
    '''
    # HWC order
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    channel = tf.shape(img)[2]
    area = tf.cast(width * height, tf.float32)

    erase_area_low_bound = tf.cast(tf.round(tf.sqrt(sl * area * r1)), tf.int32)
    erase_area_up_bound = tf.cast(tf.round(tf.sqrt((sh * area) / r1)), tf.int32)
    h_upper_bound = tf.minimum(erase_area_up_bound, height)
    w_upper_bound = tf.minimum(tf.minimum(erase_area_up_bound, width),
                               tf.cast(max_w * tf.cast(width, dtype=tf.float32), dtype=tf.int32))

    for i in range(cnt):
        h = tf.random.uniform([], erase_area_low_bound, h_upper_bound, tf.int32)
        w = tf.random.uniform([], 0, w_upper_bound, tf.int32)

        x1 = tf.random.uniform([], 0, height + 1 - h, tf.int32)
        y1 = tf.random.uniform([], 0, width + 1 - w, tf.int32)

        erase_area = tf.random.uniform([h, w, channel], 0, 0.001, tf.float32)

        erasing_img = replace_slice(img, erase_area, begin=tf.convert_to_tensor([x1, y1, 0]),
                                    size=tf.convert_to_tensor([h, w, channel]))
        # erasing_img = img[x1:x1 + h, y1:y1 + w, :].assign(erase_area)
        #erasing_img = tf.Print(erasing_img, [x1, y1, h, w], message="This is a: ")

        img = tf.cond(tf.random.uniform([], 0, 1) > probability, lambda: img, lambda: erasing_img)

    return img


def crop_pad(image, pad_x=0.07, pad_y=0.7):
    import tensorflow as tf

    size = [3000, 64]  # tf.shape(image)
    crop_x = tf.random_uniform(shape=[], minval=0,
                               maxval=pad_x)
    crop_y = tf.random_uniform(shape=[], minval=0,
                               maxval=pad_y)

    image = tf.random_crop(image, [tf.cast(tf.cast(size[0], dtype=tf.float32) * (1. - crop_y), dtype=tf.int32),
                                   tf.cast(tf.cast(size[1], dtype=tf.float32) * (1. - crop_x), dtype=tf.int32),
                                   1])

    image = tf.image.resize_image_with_crop_or_pad(image, size[0], size[1])
    # image = tf.Print(image, [tf.shape(image)], message="This is a: ")
    return image