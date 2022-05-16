#!/usr/bin/env python
# coding: utf-8

# # Setup


import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
NUM_GPUS = max(len(tf.config.experimental.list_physical_devices('GPU')), 1)


# # Trainable activation layer implementations

class LayerActivationWoSoftmax(keras.layers.Layer):
  def __init__(self, activation_functions, *args, **kwargs):
    super().__init__()
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
        initializer="uniform",
        trainable=True,
    )

  def call(self, inputs):
    outputs = []
    for activation_function in self.activation_functions:
      outputs.append(activation_function(inputs))
      
    tensor_outputs = tf.convert_to_tensor(outputs)
    multiplied_by_weights = tensor_outputs * self.w
    return tf.math.reduce_sum(multiplied_by_weights, axis=0)
  
  @classmethod
  def from_config(cls, config):
    for i, act_fn_str in enumerate(config['activation_functions']):
      config['activation_functions'][i] = keras.activations.deserialize(act_fn_str, custom_objects=vars(tf.math))
    return cls(**config)


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
    custom_objects = vars(tf.math)
    custom_objects['softmax_v2'] = keras.activations.softmax
    for i, act_fn_str in enumerate(config['activation_functions']):
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


# # Functions

# ## Dataset functions

def get_imagenet_data():
  import tensorflow_datasets as tfds
  (ds_train, ds_val), ds_info = tfds.load('imagenet2012', split=['train', 'validation'], shuffle_files=True, as_supervised=True, with_info=True, download=False)
  
  resize = lambda image, label: (tf.image.resize(image, (224, 224)), label)
  preprocess = lambda image, label: (keras.applications.imagenet_utils.preprocess_input(image, mode='torch'), label)

  ds_train = ds_train.map(resize, num_parallel_calls=tf.data.AUTOTUNE).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)    .shuffle(256 * NUM_GPUS)    .batch(256 * NUM_GPUS)    .prefetch(tf.data.AUTOTUNE)

  ds_val = ds_val.map(resize, num_parallel_calls=tf.data.AUTOTUNE).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)    .batch(256 * NUM_GPUS)    .prefetch(tf.data.AUTOTUNE)  
  return ds_train, ds_val


# ## Model functions

def get_resnet(activation_function, input_shape, classes=1000, learning_rate=0.1):
  # https://github.com/raghakot/keras-resnet/blob/master/resnet.py
  import six
  from keras.models import Model
  from keras.layers import (
      Input,
      Activation,
      Dense,
      Flatten
  )
  from keras.layers.convolutional import (
      Conv2D,
      MaxPooling2D,
      AveragePooling2D
  )
  from keras.layers.merge import add
  from keras.layers import BatchNormalization
  from keras.regularizers import l2
  from keras import backend as K


  def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return activation_function()(norm)


  def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


  def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


  def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


  def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


  def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


  def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


  def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3

  def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


  class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])

  i = keras.layers.Input(shape=input_shape)
  x = keras.layers.experimental.preprocessing.RandomRotation(factor=0.2)(i)
  x = keras.layers.experimental.preprocessing.RandomFlip()(x)
  x = ResnetBuilder.build_resnet_18(input_shape, classes)(x)
  model = keras.models.Model(i, x)
  model.compile(
      optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=['accuracy']
  )
  return model


# ## Model saving functions

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


# # Tests

from os import listdir
from os.path import isfile, join
import re
import sys

print(sys.argv)

activation_function = lambda: keras.layers.Activation('relu')
models_path = './models/resnet-imagenet-relu'
learning_rate = 0.1
if len(sys.argv) > 1 and sys.argv[1] == 'layer':
    models_path = './models/resnet-imagenet-layer-3'
    learning_rate = 0.1
    activation_function = lambda: LayerActivation(activation_functions=[
      keras.activations.relu, 
      keras.activations.tanh, 
      keras.activations.linear,
      keras.activations.sigmoid,
      keras.activations.swish,
      keras.activations.softmax
    ])
print("Using act. fn:", activation_function().get_config())
print("Using learning rate:", learning_rate)

ds_train, ds_val = get_imagenet_data()
checkpoint_callback = keras.callbacks.ModelCheckpoint(models_path + '/model-epoch-{epoch:02d}-acc-{accuracy:.3f}-val_acc-{val_accuracy:.3f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=False, mode='max', save_weights_only=False)
lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=5, min_lr=0.0001)
logger_callback = keras.callbacks.CSVLogger(models_path + '/training.log')
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=models_path + "/logs",
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq=100,
    profile_batch=100,
    embeddings_freq=0,
    embeddings_metadata=None,
)

# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

models_files = list(filter(lambda file: re.match("model-epoch-(.*)-acc-(.*)-val_acc-(.*).hdf5", file), listdir(models_path))) if os.path.isdir(models_path) else []
print("Models files:", models_files)
if models_files:
  furthest_model = max(
      models_files,
      key=lambda file: (lambda file, fn: fn(re.match("model-epoch-(.*)-acc-(.*)-val_acc-(.*).hdf5", file)))(file, lambda match: int(match.group(1)) if match is not None else -1)
  )
  epochs_ran = int(re.match("model-epoch-(.*)-acc-(.*)-val_acc-(.*).hdf5", furthest_model).group(1))
  model = load_model(models_path + '/' + furthest_model)
  print("Loaded model from file")
else:
  epochs_ran = 0
  model = get_resnet(activation_function, (224, 224, 3), learning_rate=learning_rate)
  print("Created new model")


history = model.fit(
  ds_train,
  validation_data=ds_val,
  initial_epoch=epochs_ran,
  epochs=100,
  callbacks=[lr_callback, logger_callback, tensorboard_callback, checkpoint_callback],
  steps_per_epoch=None,
  validation_steps=None
)

save_history(models_path + '/model-history.pickle', history)