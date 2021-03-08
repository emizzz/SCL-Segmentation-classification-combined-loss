import tensorflow as tf
from tensorflow.keras import backend as K



bce_loss = tf.keras.losses.BinaryCrossentropy()

def dice_coef(y_true, y_pred, smooth=1):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)

  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
  return 1 - dice_coef(y_true, y_pred, smooth=1)





def SCL_bce(y_true, y_pred, w_s=0.9, w_c=0.1):
  shape = K.shape(y_true)

  y_true_label = K.max(
    K.reshape(y_true, shape=(shape[0], shape[1] * shape[2] * shape[3])),
    axis=1
  )

  y_pred_label = K.max(
    K.reshape(y_pred, shape=(shape[0], shape[1] * shape[2] * shape[3])),
    axis=1
  )

  return (
    (w_s * bce_loss(y_true, y_pred)) + 
    (w_c * bce_loss(y_true_label, y_pred_label))
  )



def SCL_dice(y_true, y_pred, w_s=0.9, w_c=0.1):
  shape = K.shape(y_true)

  y_true_label = K.max(
    K.reshape(y_true, shape=(shape[0], shape[1] * shape[2] * shape[3])),
    axis=1
  )

  y_pred_label = K.max(
    K.reshape(y_pred, shape=(shape[0], shape[1] * shape[2] * shape[3])),
    axis=1
  )

  return (
    (w_s * dice_loss(y_true, y_pred)) +
    (w_c * bce_loss(y_true_label, y_pred_label))
  )

