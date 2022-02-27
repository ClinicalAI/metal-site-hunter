# -*- coding: utf-8 -*-

# Importing requirements

#directory for saving results
prefix="test"
save_dir = f"{prefix}/"


import os
import subprocess
import numpy as np
import random
from scipy import ndimage
from sklearn import preprocessing
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.framework.ops import Tensor
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

try:
    subprocess.run(['mkdir','-p', f'{prefix}'],check=True)
except Exception as e:
    raise


def load_data():

  data = np.load('data/train_data.npz')
  x = data["X"]
  y = data["Y"]
  y = np.reshape(y,(-1,1))

  enc = preprocessing.OneHotEncoder()
  y = enc.fit_transform(y).toarray()

  return enc,x, y


@tf.function
def rotate(volume):
    def scipy_rotate(volume):

        angles = [-15,-30,-45,-60,-90,-120,15,30,45,60,90,120,180]
        axes = [(0,1),(0,2),(1,2)]
        volume = ndimage.rotate(volume,
                                angle=random.choice(angles),
                                axes=random.choice(axes),
                                mode = 'nearest',
                                reshape=False)
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float64)
    return augmented_volume


def train_preprocessing(volume, label):
    volume = rotate(volume)
    return volume, label


def get_model_1(width=20, height=20, depth=20, channel=5):

    inputs = layers.Input((width, height, depth, channel), name='model_1')

    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu",padding='same',kernel_regularizer= tf.keras.regularizers.l2(0.00005))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu",padding='same',kernel_regularizer= tf.keras.regularizers.l2(0.00005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu",padding='same',kernel_regularizer= tf.keras.regularizers.l2(0.00005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(units=64, activation="relu",kernel_regularizer= tf.keras.regularizers.l2(0.00005))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=64, activation="relu",kernel_regularizer= tf.keras.regularizers.l2(0.00005))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=64, activation="relu",kernel_regularizer= tf.keras.regularizers.l2(0.00005))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(7, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="metal_site_model_1")
    return model


def get_model_2(width=20, height=20, depth=20, channel=5):

    inputs = layers.Input((width, height, depth, channel), name='model_2')

    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu",kernel_regularizer= tf.keras.regularizers.l2(0.0001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu",kernel_regularizer= tf.keras.regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)


    x = layers.Flatten()(x)

    x = layers.Dense(units=256, activation="relu",kernel_constraint=tf.keras.constraints.max_norm(2.))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(units=256, activation="relu",kernel_constraint=tf.keras.constraints.max_norm(2.))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(units=256, activation="relu",kernel_constraint=tf.keras.constraints.max_norm(2.))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(7, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="metal_site_model_2")
    return model

"""# Models Training"""

def train_model_1(epoch,_batch,rotation=False,prefix=''):

  _epoch = epoch
  n_splits=5
  kf = KFold(n_splits=n_splits,shuffle=False)

  i = 1

  con_mat_avg = []

  enc,x, y = load_data()
  print(enc.categories_[0])
  print(x.shape)

  for train_index, test_index in kf.split(x,y):

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))


    train_dataset = (train_loader
                     .repeat(9)
                     .shuffle(9*x_train.shape[0])
                     .map(train_preprocessing,num_parallel_calls=4)
                     .cache()
                     .batch(_batch, drop_remainder=True)
                     .prefetch(2))

    validation_dataset = (
      validation_loader
      .batch(_batch, drop_remainder=True)
      )

    model = get_model_1(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3],channel=x_train.shape[4])


    METRICS = [
                keras.metrics.TruePositives(name='tp'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalseNegatives(name='fn'),
                keras.metrics.CategoricalAccuracy(name='accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='prc', curve='PR'),
                ]

    model.compile(
                loss='categorical_crossentropy', optimizer='adam',
                metrics=METRICS)


    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+ prefix + 'model_1_epoch_{}_{}.h5'.format(_epoch,i),
                                                    monitor='val_accuracy', verbose=1,
                                                    save_best_only=True, mode='max')
    log = tf.keras.callbacks.CSVLogger(save_dir+ prefix + 'model_1_epoch_{}_{}.log'.format(_epoch,i))


    print('\n')
    print('\n')
    print('**************************************************')
    print('training on fold {} out of {}'.format(i,n_splits))
    print('**************************************************')
    print('\n')
    print('\n')


    history = model.fit(
        train_dataset,
        epochs=_epoch,
        shuffle=True,
        validation_data=validation_dataset,
        callbacks=[ checkpoint,log],
    )


    model.load_weights(save_dir+ prefix + 'model_1_epoch_{}_{}.h5'.format(_epoch,i))

    pred_score = model.predict(x_test)

    con_mat = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=pred_score.argmax(axis=1)).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_avg.append(con_mat_norm)

    labels = enc.categories_[0]

    con_mat_df = pd.DataFrame(con_mat_norm,index = labels, columns = labels)

    plt.figure()
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix fold {}'.format(i))
    plt.savefig(save_dir + prefix +f'cm_model_1_epoch_{_epoch}_fold{i}.png', dpi=600)

    i += 1


  con_mat_avg = np.array(con_mat_avg)

  con_mat_df = pd.DataFrame(con_mat_avg.mean(axis=0),index = labels, columns = labels)

  plt.figure()
  sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.title('Model 1 Confusion Matrix')
  plt.tight_layout()
  plt.savefig(save_dir + prefix +'cm_model_1_epoch_{}.png'.format(_epoch), dpi=600)


def train_model_2(epoch,_batch,rotation=False,prefix=''):

  _epoch = epoch
  n_splits=5
  kf = KFold(n_splits=n_splits,shuffle=False)

  i = 1

  con_mat_avg = []

  enc,x, y = load_data()
  print(enc.categories_[0])
  print(x.shape)

  for train_index, test_index in kf.split(x,y):

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))


    train_dataset = (train_loader
                     .repeat(9)
                     .map(train_preprocessing,num_parallel_calls=4)
                     .shuffle(9*x_train.shape[0])
                     .cache()
                     .batch(_batch, drop_remainder=True)
                     .prefetch(2))

    validation_dataset = (
      validation_loader
      .batch(_batch, drop_remainder=True)
      )

    model = get_model_2(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3],channel=x_train.shape[4])


    METRICS = [
                keras.metrics.CategoricalAccuracy(name='accuracy'),
                keras.metrics.TruePositives(name='tp'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalseNegatives(name='fn'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='prc', curve='PR'),
                ]

    model.compile(
                loss='categorical_crossentropy', optimizer='adam',
                metrics=METRICS)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+ prefix + 'model_2_epoch_{}_{}.h5'.format(_epoch,i),
                                                    monitor='val_accuracy', verbose=1,
                                                    save_best_only=True, mode='max')

    log = tf.keras.callbacks.CSVLogger(save_dir+ prefix + 'model_2_epoch_{}_{}.log'.format(_epoch,i))


    print('\n')
    print('\n')
    print('**************************************************')
    print('training on fold {} out of {}'.format(i,n_splits))
    print('**************************************************')
    print('\n')
    print('\n')

    history = model.fit(
        train_dataset,
        epochs=_epoch,
        shuffle=True,
        validation_data=validation_dataset,
        callbacks=[ checkpoint,log],
    )


    model.load_weights(save_dir+ prefix + 'model_2_epoch_{}_{}.h5'.format(_epoch,i))

    pred_score = model.predict(x_test)

    con_mat = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=pred_score.argmax(axis=1)).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_avg.append(con_mat_norm)

    labels = enc.categories_[0]

    con_mat_df = pd.DataFrame(con_mat_norm,index = labels, columns = labels)

    plt.figure()
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix fold {}'.format(i))
    plt.savefig(save_dir + prefix +f'cm_model_2_epoch_{_epoch}_fold{i}.png', dpi=600)

    i += 1

  con_mat_avg = np.array(con_mat_avg)

  con_mat_df = pd.DataFrame(con_mat_avg.mean(axis=0),index = labels, columns = labels)

  plt.figure()
  sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.title('Model 2 Confusion Matrix')
  plt.tight_layout()
  plt.savefig(save_dir + prefix +'cm_model_2_epoch_{}.png'.format(_epoch), dpi=600)

train_model_2(150,256,rotation=True,prefix=prefix)

train_model_1(150,256,rotation=True,prefix=prefix)
