import os

from keras.callbacks import Callback
import keras as K


class EvaluateInputTensor(Callback):
    """ Validate a model which does not expect external numpy data during training.
    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`.
    Instead, create a second model for validation which is also configured
    with input tensors and add it to the `EvaluateInputTensor` callback
    to perform validation.
    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.
    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, model, steps, metrics_prefix='val', verbose=2, save_path=None):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(EvaluateInputTensor, self).__init__()
        self.val_model = model
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

        self.save_path = save_path
        self.current_best = -9999

        self.best_model = None

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.model.get_weights())
        results = self.val_model.evaluate(None, None, steps=int(self.num_steps),
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)

        if self.save_path is not None:
            metrs = ""
            for i, v in logs.items():
                metrs += str(i) + '_' + str(round(v, 4)) + '_'

            if 'val_acc' in logs and self.current_best < logs['val_acc']:
                self.current_best = logs['val_acc']
                self.val_model.save(os.path.join(self.save_path, "w_%s-%s.hdf5" % (epoch, metrs)))

                self.best_model = K.models.clone_model(self.val_model)
                self.best_model.set_weights(self.val_model.get_weights())


class BestModelSaver(K.callbacks.Callback):
    def __init__(self, generator, metrics_prefix='val', verbose=2, save_path=None):
        super(BestModelSaver, self).__init__()
        self.generator = generator
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

        self.save_path = save_path
        self.current_best = -9999
        self.best_model = None

    def on_epoch_end(self, epoch, logs={}):
        results = self.model.evaluate_generator(self.generator,
                                                verbose=1, use_multiprocessing=False, workers=4)
        metrics_str = '\n'
        for result, name in zip(results, self.model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)

        if self.save_path is not None:
            metrs = ""
            for i, v in logs.items():
                metrs += str(i) + '_' + str(round(v, 4)) + '_'

            if 'val_acc' in logs and self.current_best <= logs['val_acc']:
                self.current_best = logs['val_acc']
                self.model.save(os.path.join(self.save_path, "w_%s-%s.hdf5" % (epoch, metrs)))
                # sprint('write %s' % "w_%s-%s.hdf5" %(epoch, metrs) )
                self.best_model = K.models.clone_model(self.model)
                self.best_model.set_weights(self.model.get_weights())
