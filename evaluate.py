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


def load_data_eval():

  data = np.load("data/unseen_data.npz")
  x = data["X"]
  y = data["Y"]
  y = np.reshape(y,(-1,1))

  enc = preprocessing.OneHotEncoder()
  y = enc.fit_transform(y).toarray()

  return enc,x, y


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

    outputs = layers.Dense(7, activation='softmax')(x)

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

    outputs = layers.Dense(7, activation='softmax')(x)

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

    output = layers.Dense(units=7, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=output, name='ensemble')

    return model

"""# Models Training"""

def evaluate_ensemble():

   kf = KFold(n_splits=5,shuffle=False)

   i = 1

   predictions_proba = []
   con_mat_avg = []


   _epoch = 1
   _batch = 256

   enc, x, y = load_data_eval()
   print(x.shape)

   mean_fpr = np.linspace(0, 1, 100)

   for train_index, test_index in kf.split(x,y):

     x_train, x_test = x[train_index], x[test_index]
     y_train, y_test = y[train_index], y[test_index]


     train_dataset = tf.data.Dataset.from_tensor_slices(({"ensemble_1_model_1":x_train, "ensemble_2_model_2":x_train}, y_train))
     train_dataset = train_dataset.batch(_batch, drop_remainder=True)
     train_dataset = train_dataset.prefetch(2)

     validation_dataset = tf.data.Dataset.from_tensor_slices(({"ensemble_1_model_1":x_test, "ensemble_2_model_2":x_test}, y_test))
     validation_dataset = validation_dataset.batch(_batch, drop_remainder=True).prefetch(2)

     models_list = []

     model_list_1 = [
             "/home/ahmad/REM/REM/REM-75/models/6cl_v11_/6cl_v11_model_1_epoch_150_2.h5",
             ]
     model_list_2 = [
                     "/home/ahmad/REM/REM/REM-75/models/6cl_v11_/6cl_v11_model_2_epoch_150_2.h5",
                     ]

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

     x_test_list = [ x_test for model in models_list ]

     model = ensemble(models_list)

     for layer in model.layers:
             layer.trainable = False

     METRICS = [
                 keras.metrics.TruePositives(name='tp'),
                 keras.metrics.FalsePositives(name='fp'),
                 keras.metrics.TrueNegatives(name='tn'),
                 keras.metrics.FalseNegatives(name='fn'),
                 keras.metrics.BinaryAccuracy(name='accuracy'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc'),
                 keras.metrics.AUC(name='prc', curve='PR'),
                 ]


     model.compile(
                 loss=tf.keras.losses.categorical_crossentropy,
                 optimizer='adam',
                  metrics = METRICS)

     print('\n')
     print('\n')
     print('**************************************************')
     print('training on fold {} out of {}'.format(i,5))
     print('**************************************************')
     print('\n')
     print('\n')


     model.load_weights("/home/ahmad/REM/REM/REM-75/models/6cl_v11_/6cl_v11_ensemble_epoch_500_2.h5")

     pred_score = model.predict(x_test_list)

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
     plt.title('confusion matrix fold {} (evaluation data)'.format(i))
     plt.savefig(save_dir + 'ensemble_eval_confusion_matrix_fold_{}.png'.format(i), dpi=600)

     i += 1

   con_mat_avg = np.array(con_mat_avg)

   con_mat_df = pd.DataFrame(con_mat_avg.mean(axis=0),index = labels, columns = labels)

   plt.figure()
   sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.title('Ensemble model confusion matrix (evaluation data)')
   plt.tight_layout()
   plt.savefig(save_dir + 'ensemble_eval_cm.png', dpi=600)


evaluate_ensemble()
