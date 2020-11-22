
import logging
import tensorflow as tf
from tensorflow import keras
from utils import metrics

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def train(model, optimizer, x_train, x_test, y_train, y_test, args):
    best_f1 = 0
    logger.info('Start Training!')
    model.compile(optimizer = optimizer, loss= tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy', f1_m])
    hist = model.fit(x_train, y_train, validation_data= (x_test, y_test),
              epochs= args.epochs , batch_size= 32)

    return hist

