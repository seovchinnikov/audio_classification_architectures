from keras.layers import Dense, Softmax, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Flatten, Conv2D, LSTM, \
    Reshape
import keras as K
import keras.backend as KB
import tensorflow as tf


def feature_maxpooling(num_classes):
    def frontend(x):
        x = K.layers.Dropout(rate=0.5)(x)
        x = GlobalMaxPooling2D(name='pooling')(x)

        x = Dense(512, activation='relu', name='fc1')(x)
        x = K.layers.Dropout(rate=0.5)(x)

        x = Dense(128, activation='relu', name='fc2')(x)
        x = K.layers.Dropout(rate=0.25)(x)

        x = Dense(num_classes, activation='linear')(x)
        x = Softmax()(x)

        return x

    return frontend


def feature_avgpooling(num_classes):
    def frontend(x):
        x = K.layers.Dropout(rate=0.5)(x)
        x = GlobalAveragePooling2D(name='pooling')(x)

        x = Dense(512, activation='relu', name='fc1')(x)
        x = K.layers.Dropout(rate=0.5)(x)

        x = Dense(128, activation='relu', name='fc2')(x)
        x = K.layers.Dropout(rate=0.25)(x)

        x = Dense(num_classes, activation='linear')(x)
        x = Softmax()(x)

        return x

    return frontend


def feature_avgmaxpooling(num_classes):
    def frontend(x):
        x = K.layers.Dropout(rate=0.5)(x)
        x1 = GlobalAveragePooling2D(name='pooling1')(x)
        x2 = GlobalMaxPooling2D(name='pooling2')(x)
        x = K.layers.concatenate([x1, x2], axis=-1)
        x = K.layers.Dropout(rate=0.5)(x)

        x = Dense(512, activation='relu', name='fc1')(x)
        x = K.layers.Dropout(rate=0.5)(x)

        x = Dense(128, activation='relu', name='fc2')(x)
        x = K.layers.Dropout(rate=0.5)(x)

        x = Dense(num_classes, activation='linear')(x)
        x = Softmax()(x)

        return x

    return frontend


def decision_avgpooling(num_classes):
    def frontend(x):
        x = Dense(512, activation='relu', name='fc1')(x)
        x = K.layers.Dropout(rate=0.5)(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        x = K.layers.Dropout(rate=0.5)(x)

        x = Dense(num_classes, activation='softmax')(x)
        x = GlobalAveragePooling2D(name='pooling')(x)

        return x

    return frontend


def _attention_pooling(inputs, **kwargs):
    [out, att] = inputs
    import keras as K
    epsilon = 1e-7
    att = K.backend.clip(att, epsilon, 1. - epsilon)
    normalized_att = att / K.backend.sum(att, axis=1)[:, None, :]

    res = K.backend.sum(out * normalized_att, axis=1)

    return res


def l1(x):
    import tensorflow as tf
    return tf.reduce_mean(x, axis=2, keep_dims=False)


def l2(x):
    import tensorflow as tf
    return tf.reduce_max(x, axis=2, keep_dims=False)


def sq(axis):
    def f(x):
        import tensorflow as tf
        return tf.squeeze(x, axis=axis)

    return f


def decision_attention(classes_num):
    def l1_norm(axis):
        def f(x):
            import tensorflow as tf
            return tf.norm(x, ord=1, axis=axis)

        return f

    def frontend(x):
        if len(x.shape) == 4:
            if x.shape[2] != 1:
                x1 = Lambda(l1)(x)
                x2 = Lambda(l2)(x)
                x = K.layers.Concatenate(axis=-1)([x1, x2])
                # x = K.layers.Reshape((x.shape[1], -1))(x)
            else:
                x = K.layers.Lambda(sq(2))(x)

        x = K.layers.Dropout(rate=0.5)(x)

        cla = Dense(256, activation='relu')(x)
        cla = K.layers.Dropout(rate=0.5)(cla)
        cla = Dense(128, activation='relu')(cla)
        cla = K.layers.Dropout(rate=0.5)(cla)
        cla = Dense(classes_num, activation='softmax')(cla)

        att = Dense(256, activation='relu')(x)
        att = K.layers.Dropout(rate=0.5)(att)
        att = Dense(128, activation='relu')(att)
        att = K.layers.Dropout(rate=0.5)(att)
        att = Dense(1, activation=None)(att)
        att = Softmax(axis=1)(att)
        # att = Dense(classes_num, activation='softmax')(att)

        output_layer = Lambda(
            _attention_pooling)([cla, att])

        # output_layer = Lambda(l1_norm(1))(output_layer)
        return output_layer

    return frontend


def jordi_frontend(num_classes, units_dense=200, units_cnn=128):
    def transp(axis):
        def f(x):
            import tensorflow as tf
            return tf.transpose(x, axis)

        return f

    def reduce_max(axis):
        def f(x):
            import tensorflow as tf
            return tf.reduce_max(x, axis=axis, keep_dims=False)

        return f

    def reduce_avg(axis):
        def f(x):
            import tensorflow as tf
            return tf.reduce_mean(x, axis=axis, keep_dims=False)

        return f

    def backend(x):
        ker_init = 'glorot_uniform'  # K.initializers.VarianceScaling()

        # conv layer 1 - adapting dimensions
        conv1 = Conv2D(
            filters=units_cnn,
            kernel_size=[7, int(x.shape[2])],
            padding="valid",
            activation='relu',
            kernel_initializer=ker_init)(x)

        bn_conv1 = K.layers.BatchNormalization(momentum=0.99,
                                               epsilon=1e-3)(conv1)
        bn_conv1_t = K.layers.Lambda(transp([0, 1, 3, 2]))(bn_conv1)
        bn_conv1_pad = K.layers.ZeroPadding2D(padding=(3, 0))(bn_conv1_t)

        # conv layer 2 - residual connection
        conv2 = Conv2D(
            filters=units_cnn,
            kernel_size=[7, int(bn_conv1_pad.shape[2])],
            padding="valid",
            activation='relu',
            kernel_initializer=ker_init)(bn_conv1_pad)
        conv2_t = K.layers.Lambda(transp([0, 1, 3, 2]))(conv2)
        bn_conv2 = K.layers.BatchNormalization(momentum=0.99,
                                               epsilon=1e-3)(conv2_t)
        res_conv2 = K.layers.Add()([bn_conv2, bn_conv1_t])

        # temporal pooling
        pool1 = K.layers.MaxPooling2D(pool_size=[2, 1], strides=[2, 1])(res_conv2)

        # conv layer 3 - residual connection
        bn_conv4_pad = K.layers.ZeroPadding2D(padding=(3, 0))(pool1)
        conv5 = Conv2D(
            filters=units_cnn,
            kernel_size=[7, int(bn_conv4_pad.shape[2])],
            padding="valid",
            activation='relu',
            kernel_initializer=ker_init)(bn_conv4_pad)

        bn_conv5_t = K.layers.Lambda(transp([0, 1, 3, 2]))(conv5)
        bn_conv5 = K.layers.BatchNormalization(momentum=0.99,
                                               epsilon=1e-3)(bn_conv5_t)
        res_conv5 = K.layers.Add()([bn_conv5, pool1])

        # global pooling: max and average
        max_pool2 = K.layers.Lambda(reduce_max(1))(res_conv5)
        avg_pool2 = K.layers.Lambda(reduce_avg(1))(res_conv5)
        pool2 = K.layers.Concatenate(axis=1)([max_pool2, avg_pool2])
        flat_pool2 = K.layers.Flatten()(pool2)

        # output - 1 dense layer with droupout
        flat_pool2_dropout = K.layers.Dropout(0.5)(flat_pool2)

        dense = K.layers.Dense(units=units_dense, activation='relu',
                               kernel_initializer=ker_init)(flat_pool2_dropout)
        bn_dense = K.layers.BatchNormalization(momentum=0.99,
                                               epsilon=1e-3)(dense)
        dense_dropout = K.layers.Dropout(0.5)(bn_dense)

        res = K.layers.Dense(units=num_classes, activation='softmax',
                             kernel_initializer=ker_init)(dense_dropout)
        return res

    return backend


def lstm(num_classes):
    def frontend(x):
        x = K.layers.Conv2D(128, (8, int(x.shape[2])), strides=(1, 1), activation='relu')(x)
        x = K.layers.AveragePooling2D((8, 1), strides=(8, 1))(x)

        # x = K.layers.Lambda(reshape((tf.shape(x)[0], tf.shape(x)[1], -1)))(x)
        if len(x.shape) == 4:
            if x.shape[2] != 1:
                x1 = Lambda(l1)(x)
                x2 = Lambda(l2)(x)
                x = K.layers.Concatenate(axis=-1)([x1, x2])
                # x = K.layers.Reshape((x.shape[1], -1))(x)
            else:
                x = K.layers.Lambda(sq(2))(x)

        x = K.layers.Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = K.layers.Dropout(0.5)(x)
        # x = Reshape((tf.shape(x)[1], 128))(x)
        lstm = LSTM(128, return_state=False, return_sequences=False)(x)
        dense = K.layers.Dense(units=128, activation='relu')(lstm)
        dense = K.layers.Dropout(0.5)(dense)
        res = K.layers.Dense(units=num_classes, activation='softmax')(dense)
        return res

    return frontend
