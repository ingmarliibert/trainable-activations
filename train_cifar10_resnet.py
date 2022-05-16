from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import os

class LayerActivation(keras.layers.Layer):
  def __init__(self, activation_functions, *args, **kwargs):
    super(LayerActivation, self).__init__()
    self.activation_functions = activation_functions
    self.units = len(activation_functions)
    self.softmax = keras.layers.Softmax(axis=0)

  def get_config(self):
    base_config = super().get_config()
    base_config['activation_functions'] = self.activation_functions
    return base_config

  def build(self, input_shape):
    self.w = self.add_weight(
        shape=(self.units, *([1] * len(input_shape))),
        initializer="zeros",
        trainable=True,
    )

  def call(self, inputs):
    outputs = []
    for activation_function in self.activation_functions:
      outputs.append(activation_function(inputs))
      
    tensor_outputs = tf.convert_to_tensor(outputs)
    softmax_weights = self.softmax(self.w)
    multiplied_by_weights = tensor_outputs * softmax_weights
    return tf.math.reduce_sum(multiplied_by_weights, axis=0)
  
  @classmethod
  def from_config(cls, config):
    for i, act_fn_str in enumerate(config['activation_functions']):
      custom_objects = vars(tf.math)
      custom_objects['spatial_softmax'] = spatial_softmax
      custom_objects['spatial_softmax_2'] = spatial_softmax_2
      config['activation_functions'][i] = keras.activations.deserialize(act_fn_str, custom_objects=vars(tf.math))
    return cls(**config)

class KernelActivation(keras.layers.Layer):
  def __init__(self, activation_functions, *args, **kwargs):
    super(KernelActivation, self).__init__()
    self.activation_functions = activation_functions
    self.units = len(activation_functions)
    self.softmax = keras.layers.Softmax(axis=0)

  def get_config(self):
    base_config = super(KernelActivation, self).get_config()
    base_config['activation_functions'] = self.activation_functions
    return base_config

  def build(self, input_shape):
    # Is conv layer
    if len(input_shape) > 2:
      self.w = self.add_weight(
          shape=(self.units, *([1] * len(input_shape[:-1])), input_shape[-1]),
          initializer="zeros",
          trainable=True,
      )
    # Is regular layer
    else:
      self.w = self.add_weight(
          shape=(self.units, *([1] * len(input_shape))),
          initializer="zeros",
          trainable=True,
      )

  def call(self, inputs):
    outputs = []
    for activation_function in self.activation_functions:
      outputs.append(activation_function(inputs))
      
    tensor_outputs = tf.convert_to_tensor(outputs)
    softmax_weights = self.softmax(self.w)
    multiplied_by_weights = tensor_outputs * softmax_weights
    return tf.math.reduce_sum(multiplied_by_weights, axis=0)

  @classmethod
  def from_config(cls, config):
    for i, act_fn_str in enumerate(config['activation_functions']):
      config['activation_functions'][i] = keras.activations.deserialize(act_fn_str, custom_objects=vars(tf.math))
    return cls(**config)

class NeuronActivation(keras.layers.Layer):
  def __init__(self, activation_functions, *args, **kwargs):
    super(NeuronActivation, self).__init__()
    self.activation_functions = activation_functions
    self.units = len(activation_functions)
    self.softmax = keras.layers.Softmax(axis=0)

  def get_config(self):
    base_config = super(NeuronActivation, self).get_config()
    base_config['activation_functions'] = self.activation_functions
    return base_config

  def build(self, input_shape):
    self.w = self.add_weight(
        shape=(self.units, 1, *list(map(lambda shape: shape if shape is not None else 1, input_shape[1:]))),
        initializer="zeros",
        trainable=True,
    )

  def call(self, inputs):
    outputs = []
    for activation_function in self.activation_functions:
      outputs.append(activation_function(inputs))
      
    tensor_outputs = tf.convert_to_tensor(outputs)
    softmax_weights = self.softmax(self.w)
    multiplied_by_weights = tensor_outputs * softmax_weights
    return tf.math.reduce_sum(multiplied_by_weights, axis=0)

  @classmethod
  def from_config(cls, config):
    for i, act_fn_str in enumerate(config['activation_functions']):
      config['activation_functions'][i] = keras.activations.deserialize(act_fn_str, custom_objects=vars(tf.math))
    return cls(**config)

"""# Functions

## Dataset functions
"""

def get_cifar10_preprocessed_data(n=None):
  from keras.preprocessing.image import ImageDataGenerator
  
  (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

  if n is not None:
    train_images = train_images[:n]
    train_labels = train_labels[:n]


  train_images = train_images.astype('float32')
  test_images = test_images.astype('float32')
  mean_image = np.mean(train_images, axis=0)
  train_images -= mean_image
  test_images -= mean_image
  train_images /= 128.
  test_images /= 128.

  return ((train_images, train_labels), (test_images, test_labels))

"""## Model functions"""

class BasicModel:

    """
    # Based on work by Hein, M. et al. from https://github.com/max-andr/relu_networks_overconfident/blob/master/models.py
    """

    def __init__(self, activation_function):
        self.activation_function = activation_function

    def _batch_norm(self, X):
        X = layers.BatchNormalization(momentum=0.99, epsilon=1e-5, center=True, scale=True)(X)
        return X

    def _dropout(self, X):
        X = layers.Dropout(0.5)(X)
        return X

    def _global_avg_pool(self, X):
        assert X.get_shape().ndims == 4
        return tf.reduce_mean(X, [1, 2])

    def _residual(self, X, in_filter, out_filter, stride, activate_before_residual=False):
        if activate_before_residual:
            X = self._batch_norm(X)
            X = self.activation_function()(X)
            orig_X = X
        else:
            orig_X = X
            X = self._batch_norm(X)
            X = self.activation_function()(X)

        # Sub1
        X = self._conv(X, filter_size=3, out_filters=out_filter, stride=stride)

        # Sub2
        X = self._batch_norm(X)
        X = self.activation_function()(X)
        X = self._conv(X, filter_size=3, out_filters=out_filter, stride=1)

        #Sub Add
        if in_filter != out_filter:
            orig_X = layers.AveragePooling2D(pool_size=(stride, stride), strides=(stride, stride), padding='valid')(orig_X)
            orig_X = tf.pad(orig_X, [[0, 0], [0, 0], [0, 0],
                                     [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
        X = layers.Add()([X, orig_X])
        return X

    def _conv(self, X, filter_size, stride, out_filters, biases=False):
        if biases:
            X = layers.Conv2D(filters=out_filters, kernel_size=(filter_size, filter_size),
                       strides=[stride, stride], padding='same', bias_initializer=keras.initializers.Constant(0.0),
                       kernel_regularizer=keras.regularizers.l2(0.0005),
                       bias_regularizer=keras.regularizers.l2(0.0005))(X)
        else:
            X = layers.Conv2D(filters=out_filters, kernel_size=(filter_size, filter_size),
                       strides=[stride, stride], padding='same', kernel_regularizer=keras.regularizers.l2(0.0005),
                       bias_regularizer=keras.regularizers.l2(0.0005))(X)
        return X

    def _fc_layer(self, X, n_out, bn=False, last=False):
        if len(X.shape) == 4:
            n_in = int(X.shape[1]) * int(X.shape[2]) * int(X.shape[3])
            X = layers.Flatten()(X)
        else:
            n_in = int(X.shape[1])
        X = layers.Dense(n_out, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n_in)),
                  bias_initializer=keras.initializers.Constant(0.0), kernel_regularizer=keras.regularizers.l2(0.0005),
                  bias_regularizer=keras.regularizers.l2(0.0005))(X)
        X = self._batch_norm(X) if bn else X
        if not last:
            X = self.activation_function()(X)
        else:
            X = layers.Activation('softmax')(X)
        return X

    def _conv_layer(self, X, size, n_out, stride, bn=False, biases=True):
        X = self._conv(X, size, stride, n_out, biases=biases)
        X = self._batch_norm(X) if bn else X
        X = self.activation_function()(X)
        return X


class ResNetSmall(BasicModel):

    """
    Based on implementation by Hein, M. et al. on the ResNetSmall model by from https://github.com/max-andr/relu_networks_overconfident/blob/master/models.py
    """

    def __init__(self, activation_function):
        super().__init__(activation_function=activation_function)
        self.n_filters = [16, 16, 32, 64]

    def load_model(self, input_shape, num_classes):
        strides = [1, 1, 2, 2]
        activate_before_residual = [True, False, False]
        n_resid_units = [0, 3, 3, 3]

        X_input = layers.Input(input_shape)
        X = self._conv(X_input, filter_size=3, out_filters=self.n_filters[0],
                       stride=strides[0])
        for i in range(1, len(n_resid_units)):
            X = self._residual(X, self.n_filters[i-1], self.n_filters[i], strides[i], activate_before_residual[0])
            for j in range(1, n_resid_units[i]):
                X = self._residual(X, self.n_filters[i], self.n_filters[i], 1, False)

        # Unit Last
        X = self._batch_norm(X)
        X = self.activation_function()(X)
        X = self._global_avg_pool(X)

        X = self._fc_layer(X, num_classes, last=True)

        model = keras.Model(inputs=X_input, outputs=X, name="ResNetSmall")

        return model

def get_small_resnet(activation_function, input_shape, classes=10, lr=0.01, momentum=0.9, cyclical=False, augment=False):
  if cyclical:
    print("Using cyclical lr")
    lr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=0.0001,
      maximal_learning_rate=0.5,
      scale_fn=lambda x: 1/(2.**(x-1)),
      step_size=2 * (len(get_cifar100_preprocessed_data()[0][0]) // 64)
    )
  i = keras.layers.Input(shape=input_shape)

  if augment:
    print("With augment")
    x = keras.layers.experimental.preprocessing.RandomRotation(factor=0.2)(i)
    x = keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal')(x)
    x = ResNetSmall(activation_function=activation_function).load_model(input_shape, classes)(x)
  else:
    x = ResNetSmall(activation_function=activation_function).load_model(input_shape, classes)(i)
    
  model = keras.models.Model(i, x)

  model.compile(
    optimizer=keras.optimizers.SGD(lr, momentum=momentum),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
  )

  return model

"""## Training functions"""

import json
from copy import deepcopy

def save_model(model, filepath, epoch):
  model_file = filepath.format(epoch=epoch + 1)
  print("Saving model to", model_file)
  print()
  models.save_model(model, model_file)

def model_train_regular(model, filepath, epochs, X, y, val_split=0.2, val_data=None, prune_model_epochs=[], callbacks=[], save=True, datagen=None):
  if save:
    callbacks += [keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max', save_weights_only=False)]
  if val_data is not None and val_split is None:
    print("Using given val data")
  history = model.fit(
    X, y,
    validation_split=val_split,
    validation_data=val_data,
    epochs=epochs,
    batch_size=64,
    callbacks=callbacks
  )
  return model, history


"""## Test running functions"""

import pickle

def save_history(filepath, history):
  try:
    filepath = filepath + '.history.pickle'
    print("Saving history to", filepath)
    pickle.dump(history, open(filepath, 'wb'))
  except Exception as e:
    print(e)

def load_history(filepath):
  try:
    filepath = filepath + '.history.pickle'
    return pickle.load(open(filepath, 'rb'))
  except Exception as e:
    print(e)

def load_model(filepath):
  custom_objects = {"LayerActivationWoSoftmax": LayerActivationWoSoftmax, "LayerActivation": LayerActivation, "NeuronActivation": NeuronActivation, "KernelActivation": KernelActivation}
  with keras.utils.custom_object_scope(custom_objects):
    return models.load_model(filepath)

def load_models(filepath, epochs):
  models_list = []
  for epoch in epochs:
    models_list.append(load_model(filepath.format(epoch=epoch)))
  return models_list

def load_models_from_weights(model, filepath, epochs):
  models_list = []
  for epoch in epochs:
    cloned_model = models.clone_model(model)
    cloned_model.load_weights(filepath.format(epoch=epoch))
    models_list.append(cloned_model)
  return models_list

def train_model(
    model,
    filepath_prefix, 
    filepath, 
    epochs,
    data,
    model_train_fn,
    use_test_as_val,
    save
):
  train_X, train_y = None, None
  (train_X, train_y), (test_X, test_y) = data[:2]
  val_split = 0.2
  val_data = None
  if use_test_as_val:
    val_data = (test_X, test_y)
    val_split = None
  model, history = model_train_fn(model, filepath_prefix + filepath, epochs, train_X, train_y, val_split=val_split, val_data=val_data, save=save)

  if save:
    save_history((filepath_prefix + filepath).format(epoch=0), history)
  test_loss, test_acc = model.evaluate(test_X, test_y, verbose=2)
  return model, history, test_acc, test_loss

def train(
    activation_function,
    filepath_prefix, 
    filepath, 
    epochs,
    get_data_fn,
    model_train_fn,
    get_model_fn=get_model,
    input_shape=None,
    use_test_as_val=False,
    save=True
  ):
  data = get_data_fn()
  if input_shape is None:
    input_shape = data[0][0].shape[1:]
  model = get_model_fn(activation_function=activation_function, input_shape=input_shape)
  return train_model(
      model,
      filepath_prefix, 
      filepath, 
      epochs,
      data,
      model_train_fn,
      use_test_as_val,
      save
  )

def train_or_read(
    activation_function,
    filepath_prefix, 
    filepath, 
    epochs,
    get_data_fn,
    model_train_fn,
    get_model_fn=get_model,
    input_shape=None,
    use_test_as_val=False,
    save=True
  ):
  data = get_data_fn()
  if input_shape is None:
    input_shape = data[0][0].shape[1:]
  model = get_model_fn(activation_function=activation_function, input_shape=input_shape)
  model_name = filepath.format(epoch=epochs)
  if model_name.replace("/", "") in listdir(filepath_prefix):
    print("Reading model from file")
    (train_X, train_y), (test_X, test_y) = data
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2)
    model = load_model(filepath_prefix + model_name)
    history = load_history(filepath_prefix + filepath.format(epoch=0))
    test_loss, test_acc = model.evaluate(test_X, test_y)
    return model, history, test_acc, test_loss
  else:
    print("Training model")
    return train(
      activation_function,
      filepath_prefix,
      filepath,
      epochs,
      get_data_fn,
      model_train_fn,
      get_model_fn,
      input_shape,
      use_test_as_val
    )

"""# Tests"""

from os import listdir
from os.path import isfile, join

models_path = "./models-experiments-resnet/"

"""## Tests - train data normal

### Test - layer
"""

output_train_layer = train_or_read(
    activation_function=lambda: LayerActivation(activation_functions=[
      keras.activations.relu, 
      keras.activations.tanh, 
      keras.activations.linear,
      keras.activations.sigmoid
    ]),
    filepath_prefix=models_path,
    filepath="/model-train-layer-basic-{epoch:02d}.hdf5", 
    epochs=30,
    get_data_fn=get_cifar10_preprocessed_data,
    model_train_fn=lambda *args, **kwargs: model_train_regular(*args, **kwargs, callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=5, min_lr=0.0001)]),
    get_model_fn=get_small_resnet,
    use_test_as_val=True
)

output_train_layer = train(
    activation_function=lambda: LayerActivation(activation_functions=[
      keras.activations.relu, 
      keras.activations.tanh, 
      keras.activations.linear,
      keras.activations.sigmoid,
      keras.activations.swish,
      keras.activations.softsign,
      keras.activations.softplus,
      keras.activations.selu,
      keras.activations.hard_sigmoid,
      keras.activations.gelu,
      keras.activations.elu,
    ]),
    filepath_prefix=models_path,
    filepath="/model-train-layer-normal-{epoch:02d}.hdf5", 
    epochs=30,
    get_data_fn=get_cifar10_preprocessed_data,
    model_train_fn=lambda *args, **kwargs: model_train_regular(*args, **kwargs, callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=5, min_lr=0.0001)]),
    get_model_fn=get_small_resnet,
    use_test_as_val=True
)

output_train_layer_all = train_or_read(
    activation_function=lambda: LayerActivation(activation_functions=[
      keras.activations.relu, 
      keras.activations.tanh, 
      keras.activations.linear,
      keras.activations.sigmoid,
      keras.activations.softmax,
      keras.activations.swish,
      keras.activations.softsign,
      keras.activations.softplus,
      keras.activations.selu,
      keras.activations.hard_sigmoid,
      keras.activations.gelu,
      keras.activations.elu,
      tf.math.sin,
      tf.math.cos
    ]),
    filepath_prefix=models_path,
    filepath="/model-train-layer-all-{epoch:02d}.hdf5", 
    epochs=30,
    get_data_fn=get_cifar10_preprocessed_data,
    model_train_fn=lambda *args, **kwargs: model_train_regular(*args, **kwargs, callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=5, min_lr=0.0001)]),
    get_model_fn=get_small_resnet,
    use_test_as_val=True
)

"""### Test - kernel


"""

output_train_kernel = train_or_read(
    activation_function=lambda: KernelActivation(activation_functions=[
      keras.activations.relu, 
      keras.activations.tanh, 
      keras.activations.linear,
      keras.activations.sigmoid
    ]),
    filepath_prefix=models_path,
    filepath="/model-train-kernel-basic-{epoch:02d}.hdf5", 
    epochs=30,
    get_data_fn=get_cifar10_preprocessed_data,
    model_train_fn=lambda *args, **kwargs: model_train_regular(*args, **kwargs, callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=5, min_lr=0.0001)]),
    get_model_fn=get_small_resnet,
    use_test_as_val=True
)

output_train_kernel = train_or_read(
    activation_function=lambda: KernelActivation(activation_functions=[
      keras.activations.relu, 
      keras.activations.tanh, 
      keras.activations.linear,
      keras.activations.sigmoid,
      keras.activations.swish,
      keras.activations.softsign,
      keras.activations.softplus,
      keras.activations.selu,
      keras.activations.hard_sigmoid,
      keras.activations.gelu,
      keras.activations.elu
    ]),
    filepath_prefix=models_path,
    filepath="/model-train-kernel-normal-{epoch:02d}.hdf5", 
    epochs=30,
    get_data_fn=get_cifar10_preprocessed_data,
    model_train_fn=lambda *args, **kwargs: model_train_regular(*args, **kwargs, callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=5, min_lr=0.0001)]),
    get_model_fn=get_small_resnet,
    use_test_as_val=True
)

output_train_kernel = train_or_read(
    activation_function=lambda: KernelActivation(activation_functions=[
      keras.activations.relu, 
      keras.activations.tanh, 
      keras.activations.linear,
      keras.activations.sigmoid,
      keras.activations.softmax,
      keras.activations.swish,
      keras.activations.softsign,
      keras.activations.softplus,
      keras.activations.selu,
      keras.activations.hard_sigmoid,
      keras.activations.gelu,
      keras.activations.elu,
      tf.math.sin,
      tf.math.cos
    ]),
    filepath_prefix=models_path,
    filepath="/model-train-kernel-all-{epoch:02d}.hdf5", 
    epochs=30,
    get_data_fn=get_cifar10_preprocessed_data,
    model_train_fn=lambda *args, **kwargs: model_train_regular(*args, **kwargs, callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=5, min_lr=0.0001)]),
    get_model_fn=get_small_resnet,
    use_test_as_val=True
)

"""### Test - neuron


"""

output_train_neuron = train_or_read(
    activation_function=lambda: NeuronActivation(activation_functions=[
      keras.activations.relu, 
      keras.activations.tanh, 
      keras.activations.linear,
      keras.activations.sigmoid
    ]),
    filepath_prefix=models_path,
    filepath="/model-train-neuron-basic-{epoch:02d}.hdf5", 
    epochs=30,
    get_data_fn=get_cifar10_preprocessed_data,
    model_train_fn=lambda *args, **kwargs: model_train_regular(*args, **kwargs, callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=5, min_lr=0.0001)]),
    get_model_fn=get_small_resnet,
    use_test_as_val=True
)

output_train_neuron = train_or_read(
    activation_function=lambda: NeuronActivation(activation_functions=[
      keras.activations.relu, 
      keras.activations.tanh, 
      keras.activations.linear,
      keras.activations.sigmoid,
      keras.activations.swish,
      keras.activations.softsign,
      keras.activations.softplus,
      keras.activations.selu,
      keras.activations.hard_sigmoid,
      keras.activations.gelu,
      keras.activations.elu
    ]),
    filepath_prefix=models_path,
    filepath="/model-train-neuron-normal-{epoch:02d}.hdf5", 
    epochs=30,
    get_data_fn=get_cifar10_preprocessed_data,
    model_train_fn=lambda *args, **kwargs: model_train_regular(*args, **kwargs, callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=5, min_lr=0.0001)]),
    get_model_fn=get_small_resnet,
    use_test_as_val=True
)

output_train_neuron = train_or_read(
    activation_function=lambda: NeuronActivation(activation_functions=[
      keras.activations.relu, 
      keras.activations.tanh, 
      keras.activations.linear,
      keras.activations.sigmoid,
      keras.activations.softmax,
      keras.activations.swish,
      keras.activations.softsign,
      keras.activations.softplus,
      keras.activations.selu,
      keras.activations.hard_sigmoid,
      keras.activations.gelu,
      keras.activations.elu,
      tf.math.sin,
      tf.math.cos
    ]),
    filepath_prefix=models_path,
    filepath="/model-train-neuron-all-{epoch:02d}.hdf5", 
    epochs=30,
    get_data_fn=get_cifar10_preprocessed_data,
    model_train_fn=lambda *args, **kwargs: model_train_regular(*args, **kwargs, callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=5, min_lr=0.0001)]),
    get_model_fn=get_small_resnet,
    use_test_as_val=True
)
