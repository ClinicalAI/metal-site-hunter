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
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
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
                                reshape=False)
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float64)
    return augmented_volume


def train_preprocessing(volume, label):
    volume = rotate(volume)
    return volume, label

def ensemble_preprocessing(volume,label):
    rotated_volume = rotate(volume["ensemble_1_model_1"])

    volume["ensemble_1_model_1"] = rotated_volume
    volume["ensemble_2_model_2"] = rotated_volume

    return volume,label


def get_model_1(width=20, height=20, depth=20, channel=3):

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

    outputs = layers.Dense(6, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="metal_site_model_1")
    return model


def get_model_2(width=15, height=15, depth=15, channel=1):

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

    outputs = layers.Dense(6, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="metal_site_model_2")
    return model

def ensemble (models_list):
    inputs = [model.input for model in models_list]
    outputs = [model.layers[-3].output for model in models_list]

    x = layers.Concatenate(axis=-1)(outputs)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(units=160,activation="relu",kernel_regularizer= tf.keras.regularizers.l2(0.00001))(x)
    x = layers.Dense(units=80,activation="relu",kernel_regularizer= tf.keras.regularizers.l2(0.00001))(x)
    x = layers.Dense(units=40,activation="relu",kernel_regularizer= tf.keras.regularizers.l2(0.00001))(x)
    x = layers.Dense(units=20,activation="relu",kernel_regularizer= tf.keras.regularizers.l2(0.00001))(x)
    x = layers.Dense(units=10,activation="relu",kernel_regularizer= tf.keras.regularizers.l2(0.00001))(x)
    x = layers.BatchNormalization()(x)

    x= layers.Dropout(0.3)(x)

    output = layers.Dense(units=6, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=output, name='ensemble')

    return model

"""# Models Training"""

def train_ensemble_cf(channel,epoch,_batch,rotation=False, prefix=''):

  n_splits=5
  kf = KFold(n_splits=n_splits,shuffle=False)

  i = 1

  con_mat_avg = []

  _epoch = epoch

  enc, x, y = load_data()

  for train_index, test_index in kf.split(x,y):

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]


    train_dataset = tf.data.Dataset.from_tensor_slices(({"ensemble_1_model_1":x_train,"ensemble_2_model_2":x_train }, y_train))

    train_dataset = train_dataset.repeat(9).map(ensemble_preprocessing,num_parallel_calls=4).shuffle(5*x_train.shape[0])
    train_dataset = train_dataset.cache()

    train_dataset = train_dataset.batch(_batch, drop_remainder=True)
    train_dataset = train_dataset.prefetch(2)

    validation_dataset = tf.data.Dataset.from_tensor_slices(({"ensemble_1_model_1":x_test,"ensemble_2_model_2":x_test}, y_test))

    validation_dataset = validation_dataset.batch(_batch, drop_remainder=True).prefetch(2)


    models_list = [ ]
    model_list_1 = ["/home/ahmad/REM/REM/REM-75/models/6cl_v11_/6cl_v11_model_1_epoch_150_2.h5",]
    model_list_2 = ["/home/ahmad/REM/REM/REM-75/models/6cl_v11_/6cl_v11_model_2_epoch_150_2.h5",]

    for m1 in model_list_1:
        model_1 = get_model_1(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3],channel=x_train.shape[4])
        model_1.load_weights(m1)
        models_list.append(model_1)

    for m2 in model_list_2:
        model_2 = get_model_2(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3],channel=x_train.shape[4])
        model_2.load_weights(m2)
        models_list.append(model_2)

    for model_index,model in enumerate (models_list):
        for layer in model.layers:
            layer.trainable = False
            layer._name = 'ensemble_' + str(model_index+1) + '_' + layer.name

    model = ensemble(models_list)

    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        verbose=1,
                        patience=10,
                        mode='max',
                        restore_best_weights=True)

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
    opt = tf.keras.optimizers.Adam(learning_rate=0.00005)
    model.compile(
                loss='categorical_crossentropy',
                optimizer=opt,
                 metrics = METRICS)


    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+ prefix + 'ensemble_epoch_{}_{}.h5'.format(_epoch,i),
                                                    monitor='val_accuracy', verbose=1,
                                                    save_best_only=True, mode='max')
    log = tf.keras.callbacks.CSVLogger(save_dir+ prefix + 'ensemble_epoch_{}_{}.log'.format(_epoch,i))


    print('\n')
    print('\n')
    print('**************************************************')
    print('training on fold {} out of {}'.format(i,n_splits))
    print('**************************************************')
    print('\n')
    print('\n')

    class_weight = {0: 1/0.69,
                1: 1/0.86,
                2: 1/0.7,
                3: 1/0.82,
                4: 1/0.9,
                5:1/0.95
                }

    history = model.fit(
    train_dataset,
    epochs=_epoch,
    batch_size=_batch,
    shuffle=True,
    validation_data=validation_dataset,
    callbacks=[
               checkpoint,
               log
               ],
    class_weight = class_weight
    )

    model.load_weights(save_dir+ prefix + 'ensemble_epoch_{}_{}.h5'.format(_epoch,i))

    x_test_list = [ x_test for m in models_list]

    predictions.append(model.predict(x_test_list))

    pred_score = model.predict(x_test_list)

    labels = enc.categories_[0]

    plt.figure()
    plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
    plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
    plt.plot(history.history['accuracy'], label='Accuracy (training data)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.savefig(save_dir + prefix +'ensemble_history_{}_epoch_{}.png'.format(i,_epoch), dpi=600)
    # plt.show()

    con_mat = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=pred_score.argmax(axis=1)).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_avg.append(con_mat_norm)

    con_mat_df = pd.DataFrame(con_mat_norm,index = labels, columns = labels)

    plt.figure()
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix fold {}'.format(i))
    plt.savefig(save_dir + prefix + 'ensemble_confusion_matrix_fold_{}_epoch_{}.png'.format(i,_epoch), dpi=600)
    # plt.show()

    i += 1

  con_mat_avg = np.array(con_mat_avg)

  con_mat_df = pd.DataFrame(con_mat_avg.mean(axis=0),index = labels, columns = labels)

  plt.figure()
  sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.title('Ensemble model confusion matrix'.format(channel,_epoch))
  plt.tight_layout()
  plt.savefig(save_dir + prefix + 'ensemble_cm_epoch_{}.png'.format(_epoch), dpi=600)



train_ensemble_cf(channel=False,epoch=500,_batch=256,rotation=True,prefix=prefix)
